#!/bin/bash --login
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

echo "start"
# assuming the venv being installed on the same level as 'src' package (not inside it!)
source ../../../saxs/bin/activate

python inference_demo.py