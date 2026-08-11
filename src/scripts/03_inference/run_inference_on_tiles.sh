#!/bin/bash --login
#SBATCH --job-name=infer_saxs
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00

conda activate saxs

# TODO for the user: adjust these paths
SIGNAL_DIR="FULL/PATH/TO/IMAGE/DIRECTORY/THAT/CONTAINS/SIGNAL-ONLY-TILES"
NOISE_DIR="FULL/PATH/TO/IMAGE/DIRECTORY/THAT/CONTAINS/NOISE-ONLY-TILES"

FILES_CSV_VAL_SIGNAL="PATH/TO/CSV/FILE/THAT/CONTAINS/SELECTED/FILE/NAMES/OF/SIGNAL/TILES/FOR_INFERENCE"
FILES_CSV_VAL_NOISE="PATH/TO/CSV/FILE/THAT/CONTAINS/SELECTED/FILE/NAMES/OF/NOISE/TILES/FOR_INFERENCE"

# path to the experiment output directory used during training, as this
# will contain the model checkpoints we need to load
TRAIN_OUTPUT_DIR="THE/OUTPUT/DIRECTORY/FROM/RUNNING/TRAINING/SCRIPT"
INFERENCE_OUTPUT_DIR_VAL="THE/OUTPUT/DIRECTORY/FOR/THE/INFERENCE"

#####
CKPT_FILE=$(find $TRAIN_OUTPUT_DIR -type f -name \*.ckpt | head -n 1)

GPUS=1
NUM_WORKERS=0
BATCH_SIZE=16

# NOTE: we pass options for both quantile normalization and
# fixed_constant normalization, but only the parameters are
# used for which the corresponding flag is passed as well
# (below we have --quantile_normalization and not --fixed_constant_normalization)
QUANTILE_LOW=0.01
QUANTILE_HIGH=0.9995
FIXED_CONSTANT=10000.0

TYPE="noise_remover"

python inference_on_tiles.py --signal_dir $SIGNAL_DIR\
                    --noise_dir $NOISE_DIR\
                    --files_csv_val_signal $FILES_CSV_VAL_SIGNAL\
                    --files_csv_val_noise $FILES_CSV_VAL_NOISE\
                    --output_dir $INFERENCE_OUTPUT_DIR_VAL\
                    --batch_size $BATCH_SIZE\
                    --num_workers $NUM_WORKERS\
                    --ckpt_file $CKPT_FILE\
                    --type $TYPE\
                    --quantile_normalization\
                    --normalization_quantile_low $QUANTILE_LOW\
                    --normalization_quantile_high $QUANTILE_HIGH\
                    --fixed_constant $FIXED_CONSTANT\
