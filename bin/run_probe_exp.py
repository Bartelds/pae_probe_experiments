#!/usr/bin/env python
import argparse
from collections import namedtuple
from operator import attrgetter
from pathlib import Path
import re
import sys

from joblib import delayed, parallel_backend, Parallel
import numpy as np
import pandas as pd
from pyannote.core import Segment, Timeline
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
VOCALIC = VOWELS | GLIDES | LIQUIDS | NASALS
SPEECH = STOPS | CLOSURES | FRICATIVES | VOWELS | GLIDES | LIQUIDS | \
         NASALS | OTHER
TARGETS = SPEECH

# Mapping from task names to target labels.
TASK_TARGETS = {
    'sad': SPEECH,
    'vowel': VOWELS,
    'sonorant': VOCALIC,
    'fricative': FRICATIVES}
VALID_TASK_NAMES = set(TASK_TARGETS.keys())


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


def get_classifier(clf_name, feat_dim, batch_size, weights):
    """Get classifier instance for training."""
    if clf_name not in VALID_CLASSIFIER_NAMES:
        raise ValueError(f'Unrecognized classifer "{clf_name}". '
                         f'Valid classifiers: {VALID_CLASSIFIER_NAMES}.')
    n_components = min(feat_dim, MAX_COMPONENTS)
    if clf_name == 'logistic':
        clf = LogisticRegression(class_weight='balanced')
    elif clf_name == 'max_margin':
        clf = SGDClassifier(class_weight='balanced')
    elif clf_name == 'nnet':
        # Scoring callbacks. Supported skorch callbacks:
        #
        #     https://skorch.readthedocs.io/en/stable/callbacks.html
        callbacks = [
            ('valid_precision', EpochScoring(
                'precision', lower_is_better=False, name='valid_precision')),
            ('valid_recall', EpochScoring(
                'recall', lower_is_better=False, name='valid_recall')),
            ('valid_f1', EpochScoring(
                'f1', lower_is_better=False, name='valid_f1')),
            ]

        # Gradient callbacks.
        callbacks.append(
            ('clipping', GradientNormClipping(2.0)))

        # Early stop callbacks.
        callbacks.append(
            ('EarlyStop', EarlyStopping()))

        # Instantiate our classifier.
        clf = NeuralNetClassifier(
            # Network parameters.
            MLP, module__n_hid=1,
            module__hid_dim=128,
            module__input_dim=n_components, module__n_classes=2,
            # Training batch/time/etc.
            # train_split=None,
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
    'Task', ['name', 'target_labels', 'context_size', 'classifier',
             'batch_size'])


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
    task_name = config.get('task', 'sad')
    if task_name not in VALID_TASK_NAMES:
        raise ConfigError(
            f'Encountered invalid task "{task_name}" when parsing '
            f'config file. Valid classifiers: {VALID_TASK_NAMES}')
    target_labels = TASK_TARGETS[task_name]
    task = Task(task_name, target_labels, context_size, classifier, batch_size)

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


def _get_feats_targets(utt, step, context_size, target_labels):
    # Load features from .npy file.
    feats = np.load(utt.feats_path)
    feats = add_context(feats, context_size)
    times = np.arange(len(feats))*step

    # Assign positive label to frames corresponding to target phones.
    names = ['onset', 'offset', 'label']
    segs = pd.read_csv(
        utt.phones_path, header=None, names=names, delim_whitespace=True)
    segs = segs[segs.label.isin(target_labels)]
    speech_t = Timeline(
        [Segment(seg.onset, seg.offset)
         for seg in segs.itertuples(index=False)])
    speech_t = speech_t.support()
    targets = np.zeros_like(times, dtype=np.int32)
    for seg in speech_t:
        bi, ei = np.searchsorted(times, (seg.start, seg.end))
        targets[bi:ei+1] = 1
    return feats, targets


def get_feats_targets(utterances, step, context_size, target_labels, n_jobs=1):
    """Returns features/targets for utterances.

    Parameters
    ----------
    utterances : list of Utterance
        Utterances to extract features and targets for.

    step : float
        Frame step in seconds.

    context_size : int
        Size of context window in frames.

    target_labels : iterable of str
        Labels corresponding to target classes.
    """
    target_labels = set(target_labels)
    with parallel_backend('multiprocessing', n_jobs=n_jobs):
        f = delayed(_get_feats_targets)
        res = Parallel()(
            f(utterance, step, context_size, target_labels)
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
        '--n-jobs', nargs=None, default=1, type=int, metavar='JOBS',
        help='number of parallel jobs (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # Load config.
    task, train_dsets, test_dsets = load_task_config(args.config)

    print('Training classifiers...')
    models = {}
    for dset in train_dsets:
        print(f'Training classifier for dataset "{dset.name}"...')

        # Load appropriate training set.
        feats, targets = get_feats_targets(
            dset.utterances, dset.step, task.context_size, task.target_labels,
            args.n_jobs)
        n_frames, feat_dim = feats.shape
        print(f'FRAMES: {n_frames}, DIM: {feat_dim}')

        # Fit classifier.
        pos_freq = targets.mean()
        weights = np.array([1-pos_freq, pos_freq], dtype=np.float32)
        weights = torch.from_numpy(1/weights)
        weights /= weights.sum()
        clf = get_classifier(
            task.classifier, feat_dim, task.batch_size, weights)
        print('Fitting...')
        clf.fit(feats, targets)
        models[dset.name] = clf

    print('Testing...')
    test_data = {}
    for dset in test_dsets:
        feats, targets = get_feats_targets(
            dset.utterances, dset.step, task.context_size, task.target_labels,
            args.n_jobs)
        test_data[dset.name] = {
            'feats': feats,
            'targets': targets}

    records = []
    for train_dset_name in sorted(models):
        clf = models[train_dset_name]
        for test_dset_name in test_data:
            feats = test_data[test_dset_name]['feats']
            targets = test_data[test_dset_name]['targets']
            preds = clf.predict(feats)

            acc = metrics.accuracy_score(targets, preds)
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                targets, preds, pos_label=1, average='binary')
            records.append({
                'train': train_dset_name,
                'test': test_dset_name,
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1})
    scores_df = pd.DataFrame(records)
    print(scores_df)


if __name__ == '__main__':
    main()
