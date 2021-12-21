#!/usr/bin/env python
"""Run binary clasification probing experiment."""
import argparse
from collections import namedtuple
from operator import attrgetter
from pathlib import Path
import re
import sys

from joblib import delayed, parallel_backend, Parallel
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, GradientNormClipping, EarlyStopping
import torch
from torch import nn
import yaml

torch.multiprocessing.set_sharing_strategy('file_system')


Utterance = namedtuple(
    'Utterance', ['uri', 'feats_path', 'phones_path'])

STOPS = {'p', 't', 'k',
         'b', 'd', 'g'}
CLOSURES = {'pcl', 'tcl', 'kcl',
            'bcl', 'dcl', 'gcl'}
FRICATIVES = {'ch', 'th', 'f', 's', 'sh',
              'jh', 'dh', 'v', 'z', 'zh',
              'hh'}
VOWELS = {'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
          'eh', 'el', 'em', 'en', 'eng', 'er', 'ey', 'ih', 'ix',
          'iy', 'ow', 'oy', 'uh', 'uw', 'ux'}
GLIDES = {'w', 'y'}
LIQUIDS = {'l', 'r'}
NASALS = {'m', 'n', 'ng', 'nx'}
OTHER = {'dx', 'hv', 'q'}
SILENCE = {'sil'}
VOCALIC = VOWELS | GLIDES | LIQUIDS | NASALS
SPEECH = STOPS | CLOSURES | FRICATIVES | VOWELS | GLIDES | LIQUIDS | \
         NASALS | OTHER
PHONES = SPEECH | SILENCE


# TRAIN/TEST: use original phone set
# EVAL: convert predicted/reference labels to reduced set

# Phones remapped from original TIMIT set to the 39 phone set usually used
# when scoring PER/classification accuracy (e.g., in Kaldi recipes).
TIMIT_39_REMAPS = {
    'ao' : 'aa',
    'ax' : 'ah',
    'ax-h' : 'ah',
    'axr' : 'er',
    'bcl' : 'sil',
    'dcl' : 'sil',
    'el' : 'l',
    'em' : 'm',
    'en' : 'n',
    'eng' : 'ng',
    'epi' : 'sil',
    'gcl' : 'sil',
    'h#' : 'sil',
    'hv' : 'hh',
    'ix' : 'ih',
    'kcl' : 'sil',
    'nx' : 'n',
    'pau' : 'sil',
    'pcl' : 'sil',
    'q' : 'sil',
    'tcl' : 'sil',
    'ux' : 'uw',
    'zh' : 'sh'}

PHONES39 = {TIMIT_39_REMAPS.get(phone, phone) for phone in PHONES}


# Mapping from binary classification task names to target labels.
TASK_TARGETS = {
    'sad': SPEECH,
    'vowel': VOWELS,
    'sonorant': VOCALIC,
    'fricative': FRICATIVES}
VALID_TASK_NAMES = set(TASK_TARGETS.keys()) | {'phone'}


def get_class_mapping(phones, target_phones=None):
    """Return mapping from phones to integer ids of corresponding classes.

    If ``target_phones`` is specified, maps the elements of ``target_phones``
    to 1 and all other phones to 0. Otherwise, returns a bijection between
    ``phones`` and ``range(len(phones))``.

    Parameters
    ----------
    phones : iterable of str
        Phones.

    target_phones : iterable of str, optional
        All phones in ``target_phones`` will be mapped to 1. All other phones
        to 0.
        (Default: None)

    Returns
    -------
    phone_to_id : dict
        Mapping from phones to non-negative integer ids.
    """
    if target_phones is None:
        return {phone:n for n, phone in enumerate(sorted(phones))}
    phone_to_id = {}
    for phone in sorted(phones):
        phone_to_id[phone] = 1 if phone in target_phones else 0
    return phone_to_id


# TRAIN/TEST: use original phone set
# EVAL: convert predicted/reference labels to reduced set



class MLP(nn.Module):
    def __init__(self, input_dim, n_hid=1, hid_dim=512, n_classes=2,
                 dropout=0.5):
        super(MLP, self).__init__()
        components = []
        sizes = [input_dim] + [hid_dim]*n_hid
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            components.append(nn.Linear(in_dim, out_dim))
            components.append(nn.ReLU())
            components.append(nn.Dropout(dropout))
        components.append(nn.Linear(hid_dim, n_classes))
        self.logits = nn.Sequential(*components)

    def forward(self, X, **kwargs):
        X = self.logits(X)
        return X


VALID_CLASSIFIER_NAMES = {'logistic', 'max_margin', 'nnet'}
MAX_COMPONENTS = 400  # Keep at most this many components after SVD.


def get_classifier(clf_name, feat_dim, batch_size, n_classes, weights,
                   sgd_kwargs={}):
    """Get classifier instance for training."""
    if clf_name not in VALID_CLASSIFIER_NAMES:
        raise ValueError(f'Unrecognized classifer "{clf_name}". '
                         f'Valid classifiers: {VALID_CLASSIFIER_NAMES}.')
    n_components = min(feat_dim, MAX_COMPONENTS)
    if clf_name == 'logistic':
        clf = LogisticRegression(class_weight='balanced')
    elif clf_name == 'max_margin':
        clf = SGDClassifier(class_weight='balanced', **sgd_kwargs)
    elif clf_name == 'nnet':
        # Scoring callbacks for binary clasification tasks.
        callbacks = []
        if n_classes == 2:
            callbacks.append(
                ('valid_precision',
                 EpochScoring('precision', lower_is_better=False,
                              name='valid_precision')))
            callbacks.append(
                ('valid_recall',
                 EpochScoring('recall', lower_is_better=False,
                              name='valid_recall')))
            callbacks.append(
                ('valid_f1',
                 EpochScoring('f1', lower_is_better=False, name='valid_f1')))

        # Clip gradients to L2-norm of 2.0
        callbacks.append(
            ('clipping', GradientNormClipping(2.0)))

        # Allow early stopping.
        callbacks.append(
            ('EarlyStop', EarlyStopping()))

        # Instantiate our classifier.
        clf = NeuralNetClassifier(
            # Network parameters.
            MLP, module__n_hid=1, module__hid_dim=128,
            module__input_dim=n_components, module__n_classes=n_classes,
            # Training batch/time/etc.
            max_epochs=50, batch_size=batch_size,
            # Training loss.
            criterion=nn.CrossEntropyLoss,
            criterion__weight=weights,
            # Optimization parameters.
            optimizer=torch.optim.Adam, lr=3e-4,
            # Parallelization.
            iterator_train__shuffle=True,
            iterator_train__num_workers=4,
            iterator_valid__num_workers=4,
            # Scoring callbacks.
            callbacks=callbacks)

        # Ensure ANSI escape sequences (e.g., colors) are stripped from log
        # output before printing. Ensures output is clean if redirected to
        # file.
        def print_scrubbed(txt):
            txt = re.sub(r'\x1b\[\d+m', '', txt)
            print(txt)
        clf.set_params(callbacks__print_log__sink=print_scrubbed)
    clf = Pipeline([
        ('scaler', TruncatedSVD(n_components=n_components)),
        ('clf', clf)])
    return clf


def load_utterances(uris_file, feats_dir, phones_dir):
    """Return utterances corresponding to partition."""
    uris_file = Path(uris_file)
    feats_dir = Path(feats_dir)
    phones_dir = Path(phones_dir)

    # Load URIs for utterances.
    with open(uris_file, 'r') as f:
        uris = {line.strip() for line in f}

    # Check for corresponding .npy/.lab files.
    utterances = []
    for uri in uris:
        feats_path = Path(feats_dir, uri + '.npy')
        phones_path = Path(phones_dir, uri + '.lab')
        if not feats_path.exists() or not phones_path.exists():
            continue
        utterances.append(
            Utterance(uri, feats_path, phones_path))

    return utterances


# To distinguish from skorch.dataset.Dataset
Datasets = namedtuple(
    'Dataset', ['name', 'utterances', 'step'])

Task = namedtuple(
    'Task', ['name', 'phone_to_id', 'id_to_phone', 'context_size',
             'classifier', 'batch_size'])


class ConfigError(Exception):
    pass


def load_task_config(fn):
    """Load task from configuration file."""
    fn = Path(fn)
    with open(fn, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Batch size for neural network training.
    batch_size = config.get('batch_size', 128)

    # Context window size in frames.
    context_size = config.get('context_size', 0)

    # Classifier type.
    classifier = config.get('classifier', 'logistic')
    if classifier not in VALID_CLASSIFIER_NAMES:
        raise ConfigError(
            f'Encountered invalid classifier "{classifier}" when parsing '
            f'config file. Valid classifiers: {VALID_CLASSIFIER_NAMES}')

    # Task.
    task_name = config.get('task', None)
    if task_name not in VALID_TASK_NAMES:
        raise ConfigError(
            f'Encountered invalid task "{task_name}" when parsing '
            f'config file. Valid classifiers: {VALID_TASK_NAMES}')
    target_phones = TASK_TARGETS.get(task_name, None)
    phone_to_id = get_class_mapping(PHONES, target_phones)
    id_to_phone = {n:phone for phone, n in phone_to_id.items()}
    task = Task(task_name, phone_to_id, id_to_phone, context_size, classifier,
                batch_size)

    # Load partitons.
    def _load_dsets(d, test=False):
        dsets = []
        for dset_name in d:
            dset = d[dset_name]
            utterances = load_utterances(
                dset['uris'], dset['feats'], dset['phones'])
            if test:
                utterances.sort(key=attrgetter('uri'))
            dsets.append(
                Datasets(dset_name, utterances, dset['step']))
        return dsets
    train_dsets = _load_dsets(config['train_data'])
    test_dsets = _load_dsets(config['test_data'], test=True)
    return task, train_dsets, test_dsets


def _get_feats_targets(utt, step, context_size, phone_to_id):
    # Load features from .npy file.
    feats = np.load(utt.feats_path)
    feats = add_context(feats, context_size)
    times = np.arange(len(feats))*step

    # Load segments.
    names = ['onset', 'offset', 'label']
    segs = pd.read_csv(
        utt.phones_path, header=None, names=names, delim_whitespace=True)

    # Convert to frame-level labels.
    targets = np.zeros_like(times, dtype=np.int32)
    for seg in segs.itertuples(index=False):
        bi, ei = np.searchsorted(times, (seg.onset, seg.offset))
        targets[bi:ei+1] = phone_to_id[seg.label]
    return feats, targets


def get_feats_targets(utterances, step, context_size, phone_to_id, n_jobs=1):
    """Returns features/targets for utterances.

    Parameters
    ----------
    utterances : list of Utterance
        Utterances to extract features and targets for.

    step : float
        Frame step in seconds.

    context_size : int
        Size of context window in frames.

    phone_to_id : dict
        Mapping from phone to integer ids.

    n_jobs : int, optional
        Number of parallel jobs to use,
    """
    with parallel_backend('multiprocessing', n_jobs=n_jobs):
        f = delayed(_get_feats_targets)
        res = Parallel()(
            f(utterance, step, context_size, phone_to_id)
            for utterance in utterances)
    feats, targets = zip(*res)

    # Garbage collection
    feats_tmp = np.concatenate(feats, axis=0).astype(np.float32)
    del feats
    feats = feats_tmp
    targets_tmp = np.concatenate(targets, axis=0).astype(np.int64)
    del targets
    targets = targets_tmp

    return feats, targets


def add_context(feats, win_size):
    """Append context to each frame.

    Parameters
    ----------
    feats : ndarray, (n_frames, feat_dim)
        Features.

    win_size : int
        Number of frames on either side to append.

    Returns
    -------
    ndarray, (n_frames, feat_dim*(win_size*2 + 1))
        Features with context added.
    """
    if win_size <= 0:
        return feats
    feats = np.pad(feats, [[win_size, win_size], [0, 0]], mode='edge')
    inds = np.arange(-win_size, win_size+1)
    feats = np.concatenate(
        [np.roll(feats, ind, axis=0) for ind in inds], axis=1)
    feats = feats[win_size:-win_size, :]
    return feats


def main():
    parser = argparse.ArgumentParser(
        description='run binary classification probes', add_help=True)
    parser.add_argument(
        'config', type=Path, help='path to task config')
    parser.add_argument(
        '--seed', metavar='SEED', default=11238421, type=int,
        help='seed for RNG')
    parser.add_argument(
        '--n-jobs', nargs=None, default=1, type=int, metavar='JOBS',
        help='number of parallel jobs (default: %(default)s)')
    parser.add_argument(
        '--preds', default=False, action="store_true",
        help='get labels of predictions')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Loading task config...')
    task, train_dsets, test_dsets = load_task_config(args.config)

    print('Training classifiers...')
    models = {}
    for dset in train_dsets:
        print(f'Training classifier for dataset "{dset.name}"...')

        # Load appropriate training set.
        feats, targets = get_feats_targets(
            dset.utterances, dset.step, task.context_size, task.phone_to_id,
            args.n_jobs)
        n_frames, feat_dim = feats.shape

        # Fit classifier.
        weights = (1 / np.bincount(targets)).astype(np.float32)
        n_classes = weights.size
        weights[weights == np.inf] = 0
        weights = torch.from_numpy(weights)
        weights /= weights.sum()
        sgd_kwargs = {'n_jobs' : args.n_jobs}
        if task.name == 'phones':
            sgd_kwargs = {'tol' : 1e-4,
                          'early_stopping' : True,
                          'validation_fraction' : 0.2 }
        clf = get_classifier(
            task.classifier, feat_dim, task.batch_size, n_classes, weights,
            sgd_kwargs)

        print('Fitting...')
        clf.fit(feats, targets)
        models[dset.name] = clf

    print('Testing...')
    test_data = {}
    for dset in test_dsets:
        feats, targets = get_feats_targets(
            dset.utterances, dset.step, task.context_size, task.phone_to_id,
            args.n_jobs)
        test_data[dset.name] = {
            'feats': feats,
            'targets': targets}

    records = []
    for train_dset_name in sorted(models):
        clf = models[train_dset_name]
        for test_dset_name in test_data:
            # Predict frame-level classes.
            feats = test_data[test_dset_name]['feats']
            targets = test_data[test_dset_name]['targets']
            preds = clf.predict(feats)

            # Calculate accuracy, precision, recall, and F1.
            if task.name != 'phone':
                # For binary classification tasks, we just care about
                # precision, recall, F1 for target class (e.g., speech,
                # sonorants, fricatives).
                acc = metrics.accuracy_score(targets, preds)
                precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                    targets, preds, pos_label=1, average='binary')
            else:
                # For phone classification, we compute macro-averaged precision,
                # recall, and F1 using the standard 39-phone reduction of the
                # TIMIT phone set. Since our original classifier was trained
                # using the full set, we need to remap both the reference
                # and system labels.
                def _to_timit39(ids):
                    phones = [task.id_to_phone[id] for id in ids]
                    phones = [TIMIT_39_REMAPS.get(phone, phone)
                              for phone in phones]
                    return phones
                targets = _to_timit39(targets)
                preds = _to_timit39(preds)
                acc = metrics.accuracy_score(targets, preds)
                precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                    targets, preds, average='weighted')
                if args.preds:
                    print(f'\nPredictions {task.classifier} classifier for dataset "{dset.name}"...')

                    byPhone_precision, byPhone_recall, byPhone_f1, byPhone_support = metrics.precision_recall_fscore_support(targets, preds, labels=list(set(targets)), average=None)
                    labels = list(set(targets))

                    byPhone_results = pd.DataFrame(
                        {'label': labels, 'precision': byPhone_precision, 'recall': byPhone_recall, 'f1': byPhone_f1, 'support': byPhone_support}
                    )

                    print(byPhone_results.to_string())

            # Update dataframe.
            records.append({
                'train': train_dset_name,
                'test': test_dset_name,
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1})
    scores_df = pd.DataFrame(records)
    scores_df = scores_df[
        ['train', 'test', 'acc', 'precision', 'recall', 'f1']]
    print(f"\n{scores_df}")


if __name__ == '__main__':
    main()
