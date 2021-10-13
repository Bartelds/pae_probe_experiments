Running experiments using wav2vec 2.0 features
==============================================
Instructions
-------------
- Representations are automatically computed for each of the transformer layers (edit layer variable in ``run.sh`` to change this).
- Representations are evaluated using the phone classification task utilizing the standard TIMIT to shorten runtime.
- Run ``run.sh``.
-------------
Results and plots
-------------
- Results for phone classification on TIMIT using representations from [wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h) and [wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) are in the `logs/` and `plots/` directories.