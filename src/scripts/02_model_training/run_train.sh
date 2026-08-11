#!/bin/bash --login
#SBATCH --job-name=train_noise_remover_saxs_tiles
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00

conda activate saxs

SEED=42

# %% The user shall adjust these paths and names

BASE_DIR="/home/smid55/10403_data/exportB/"
SIGNAL_DIR=$BASE_DIR/signal_tiles
NOISE_DIR=$BASE_DIR/noise_tiles

FILES_CSV_TRAIN_SIGNAL=$BASE_DIR/signal_train_tiles.csv
FILES_CSV_TRAIN_NOISE=$BASE_DIR/noise_train_tiles.csv

FILES_CSV_VAL_SIGNAL=$BASE_DIR/signal_val_tiles.csv
FILES_CSV_VAL_NOISE=$BASE_DIR/noise_val_tiles.csv




OUTPUT_DIR="YOUR/OUTPUT/DIRECTORY"

# the hyperparameters we trained our models with
BATCH_SIZE=16
NUM_WORKERS=0
GPUS=1
EPOCHS=400
LEARNING_RATE=1.e-4
LOSS="mae"
MODEL="equnet32"
TYPE="noise_remover"

OUTPUT_DIR="$BASE_DIR/$TYPE/10403_B"


WEIGHT_RECO=1.
WEIGHT_NOISE=1.

# NOTE: we pass options for both quantile normalization and
# fixed_constant normalization, but only the parameters are
# used for which the corresponding flag is passed as well
# (below we have --quantile_normalization and not --fixed_constant_normalization)
QUANTILE_LOW=0.01
QUANTILE_HIGH=0.9995
FIXED_CONSTANT=10000.0

python train.py --signal_dir $SIGNAL_DIR\
                --noise_dir $NOISE_DIR\
                --files_csv_train_signal $FILES_CSV_TRAIN_SIGNAL\
                --files_csv_train_noise $FILES_CSV_TRAIN_NOISE\
                --files_csv_val_signal $FILES_CSV_VAL_SIGNAL\
                --files_csv_val_noise $FILES_CSV_VAL_NOISE\
                --seed $SEED\
                --default_root_dir $OUTPUT_DIR\
                --batch_size $BATCH_SIZE\
                --num_workers $NUM_WORKERS\
                --max_epochs $EPOCHS\
                --log_every_n_steps 1\
                --learning_rate $LEARNING_RATE\
                --num_best_checkpoints 3\
                --checkpoint_every_n_epochs 1\
                --loss $LOSS\
                --accelerator "gpu"\
                --devices $GPUS\
                --model $MODEL\
                --type $TYPE\
                --weight_reco $WEIGHT_RECO\
                --weight_noise $WEIGHT_NOISE\
                --normalization_quantile_low $QUANTILE_LOW\
                --normalization_quantile_high $QUANTILE_HIGH\
                --fixed_constant $FIXED_CONSTANT\
                --quantile_normalization\
