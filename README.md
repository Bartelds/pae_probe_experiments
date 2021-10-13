## Overview
This repository contains code for replicating the results from the following paper:

- Ma, Danni, Ryant, Neville, and Liberman, Mark. (2021). "Probing acoustic representations for phonetic properties." Proceedings of ICASSP. https://arxiv.org/abs/2010.13007

To use representations from hidden wav2vec2.0 transformer layers, this repository is updated with the code from Bartelds, M., de Vries, W., Sanal, F., Richter, C., Liberman, M., & Wieling, M. (2020). "Neural Representations for Modeling Variation in English Speech". https://arxiv.org/abs/2011.12649

## Installation
1. Create and activate a new virtual environment.
2. Clone the repo:

       git clone https://github.com/Bartelds/pae_probe_experiments.git
       cd pae_probe_experiments/
	
3. Install the required Python packages:

       pip install -r requirements
## Preparing the data
In order to run these experiments, you first need to obtain and process the various TIMIT corpora. Instructions for obtaining the corpora from LDC and running the processing pipeline are contained in `data/README.md`.

## Running the experiments
In the paper we report results for two conventional (MFCCs and mel filterbanks) acoustic representations and 5 pre-trained acoustic representations:

* [wav2vec](https://arxiv.org/abs/1904.05862)
* [vq-wav2vec](https://arxiv.org/abs/1910.05453)
* [Mockingjay](https://arxiv.org/abs/1910.12638)
* [DeCoAR](https://arxiv.org/abs/1912.01679)
* [wav2vec2.0](https://arxiv.org/abs/2006.11477)

Code for reproducing our published results for each representation is located under `experiments/`, which contains one sub-directory per representation. To run the experiments, change to the desired directory and follow the instructions in the `README`. When the run is finished, the results (output to STDOUR and STDERR) for each run will be saved to the `logs/` directory.

For instance, to reproduce the results for wav2vec:

	cd experiments/wav2vec
	./run.sh

This will download the pre-trained [Fairseq](https://github.com/pytorch/fairseq) wav2vec model, extract frame-level features for each TIMIT variant, and run the full set of in-domain and cross-domain experiments for each probing task. Results will be stored to `logs/`; e.g.:

    logs/extract_wav2vec-large_ctimit.stderr
    logs/extract_wav2vec-large_ctimit.stdout
    logs/extract_wav2vec-large_ffmtimit.stderr
    logs/extract_wav2vec-large_ffmtimit.stdout
    ...
    logs/fricative_logistic.stderr
    logs/fricative_logistic.stdout
    logs/fricative_max_margin.stderr
    logs/fricative_max_margin.stdout
    ...


## Reproducibility

Due to differing seeds to the NumPy and Torch RNGs, running the code in this repo will not exactly duplicate the results reported in the ICASSP paper. However, the obtained results should be within 1-2% on average.

## Citing
If you use the code in your work, please cite the associated ICASSP paper:

```
@inproceedings{ma2021probing,
 title={Probing Acoustic Representations for Phonetic Properties},
 author={Ma, Danni and Ryant, Neville and Liberman, Mark},
 booktitle={Proceedings of ICASSP},
 eprint={2010.13007},
 archivePrefix={arXiv},
 primaryClass={eess.AS},
 year={2021}
}
```
