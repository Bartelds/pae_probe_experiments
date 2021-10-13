#!/usr/bin/env python
"""Generate configuration files needed for experimnts."""
import argparse
from pathlib import Path
import sys

import yaml


# PROBING_TASKS = ['sad', 'vowel', 'sonorant', 'fricative', 'phone']
PROBING_TASKS = ['phone']
CLASSIFIERS = ['logistic', 'max_margin', 'nnet']
# CORPORA = ['ctimit', 'ffmtimit', 'ntimit', 'stctimit', 'timit', 'wtimit']
CORPORA = ['timit']


def main():
    parser = argparse.ArgumentParser(
        'generate configuration files', add_help=True)
    parser.add_argument(
        '--feats_dir', metavar='feats-dir', type=Path,
        help='directory of extracted features')
    parser.add_argument(
        '--model', help='name of model')
    parser.add_argument(
        '--config_dir', metavar='config-dir', type=Path,
        help='output directory for config files')
    parser.add_argument(
        '--data_dir', metavar='data-dir', type=Path,
        help='directory containing processed TIMIT variants')
    parser.add_argument(
        '--context-size', metavar='CONTEXT', default=0, type=int,
        help='include CONTEXT frames on either side of each frame as '
             'context (default: %(default)s)')
    parser.add_argument(
        '--batch-size', metavar='BATCH', default=1024, type=int,
        help='when training MLP, use batches of BATCH examples '
             '(default: %(default)s)')
    parser.add_argument(
        '--step', metavar='SECONDS', default=0.01, type=float,
        help='frame step size in seconds (default: %(default)s)')
    parser.add_argument(
        '--layer', default=24, type=int,
        help='select layer to process')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    args.config_dir.mkdir(parents=True, exist_ok=True)

    # Determine parameters for configuration files.
    data = []
    for task in PROBING_TASKS:
        for clf in CLASSIFIERS:
            train_data = {}
            test_data = {}
            for corpus in CORPORA:
                feats_dir = Path(args.feats_dir, args.model, 'layer-' + str(args.layer))
                phones_dir = Path(args.data_dir, corpus, 'phones')
                train_data[corpus] = {
                    'uris' : str(args.data_dir / 'lists/train.ids'),
                    'step' : args.step,
                    'feats' : str(feats_dir),
                    'phones' : str(phones_dir)}
                test_data[corpus] = {
                    'uris' : str(args.data_dir / 'lists/test_full.ids'),
                    'step' : args.step,
                    'feats' : str(feats_dir),
                    'phones' : str(phones_dir)}
            data = {
                'task' : task,
                'classifier' : clf,
                'context_size' : args.context_size,
                'batch_size' : args.batch_size,
                'train_data' : train_data,
                'test_data' : test_data}
            yaml_path = args.config_dir / f'{task}_{clf}.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)


if __name__ == '__main__':
    main()
