#!/bin/bash

set -e  # Exit on error

DATA_DIR=/home/pae_probe_experiments/data/
FEATS_DIR=/data/probing/feats/
MODEL_NAMES="wav2vec2-large-960h wav2vec2-large-xlsr-53"
LAYER="all"

##############################################################################
# Configuration
##############################################################################
nj=-1   # Number of parallel jobs for CPU operations.
stage=0

. path.sh

mkdir -p logs/


##############################################################################
# Extract features
##############################################################################
if [ $stage -le 0 ]; then
	for model in $MODEL_NAMES; do
		python3 ../../bin/gen_wav2vec_feats_hf.py -i $DATA_DIR/timit/wav/ -o $FEATS_DIR/{model}/layer-{layer}/ -m $model -l $LAYER
	done
fi

##############################################################################
# Run classification tasks.
##############################################################################
if [ $stage -le 1 ]; then
    for model in $MODEL_NAMES; do
    	for i in $(seq 1 3 24); do
    		echo "Pocessing features from $model layer $i"
			echo "$0: Preparing config files..."
    		# NOTE: Wav2vec 2.0 uses step size of 20 ms.
    		python3 ../../bin/gen_config_files.py \
			--step 0.020 \
        		--feats_dir $FEATS_DIR --model $model --config_dir configs/tasks --data_dir $DATA_DIR --layer $i

    		echo "$0: Running classification experiments..."
    		for config in `ls configs/tasks/*.yaml`; do
        		bn=${config##*/}
        		name=${bn%.yaml}
        		echo $name
				echo "$model layer $i" >> logs/${model}_${name}.stdout
        		python3 ../../bin/run_probe_exp.py \
            		--n-jobs $nj $config \
            		>> logs/${model}_${name}.stdout \
            		2>> logs/${model}_${name}.stderr &
    		done
    		wait
    	done
    done
fi
