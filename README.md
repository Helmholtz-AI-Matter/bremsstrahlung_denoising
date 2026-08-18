# Bremsstrahlung Background Removal from HIBEF based SAXS data

## Project Info


## Abstract / Summary

Large amounts of energetic bremsstrahlung are created in experiments where ultra high intensity laser interacts with solid targets. This constitutes a distinct background signal on detectors used in the experiment. Such background then can obscure the desired measured signal. We provide a tool which can distinguish this background from the useful signal. The first and prominent case where this was utilized is in the detection of Small angle x-ray scattering (SAXS) diagnostics at the HED instrument at European XFEL, but we believe this tool could find much broader usage.

## Workflow

#### 1) Processing the data
1) Get the experimental data
    - That is, just get two folders with tiff files for the *preshot* and *main shot* files.
2) Create the tiles
    - Use the script **src/scripts/01_data_preprocessing/patchwork.py** to create noise and signal tiles.
    - By default, those are created in export/noise_tiles and export/signal_tiles

### 2) Creating the neural network

1) Train the network:
	- Execute the script **src/scripts/02_model_training/run_train.sh** or create a copy of it if you want to experiment with different configurations.
    - run it on some cluster via **sbatch run_train.sh**


### 3) Apply the model to experimental data

1) Check out the **src/scripts/03_inference/inference_demo.py** to see how predictions can be made using an existing trained network.
We have made our trained model available at https://doi.org/10.14278/rodare.4956.

## Code information

### Installation and Setup

1. Create a virtual environment for this project, e.g. using anaconda (python virtualenvs are also fine) via

```
uv venv saxs python 3.10
```

2. Once the environment got created, activate it with
```
source saxs/bin/activate
```
The commandline prompt should now have changed slightly to read
```
(saxs) yourname@yourcomputer:~$
```
indicating that the environment is active.

3. Clone the repository for this project via
```

# using ssh which requires that you have registered your ssh key on the github page
git clone git@github.com:Helmholtz-AI-Matter/bremsstrahlung_denoising.git
``````

or, in case ssh has not been set up with github via
```
git clone https://github.com/Helmholtz-AI-Matter/bremsstrahlung_denoising.git
```

4. Before continuing to install, make sure that a C compiler (like gcc) is available on your system, as some dependencies require this (like `py3nj` which is installed with the `escnn` library).

5. Install dependencies and the project itself (note that this will install the dependencies into your virtual environment only, not systemwide) as follows:

5.1 Install our package

```
cd src
# installs dependencies
uv pip install -r requirements.txt
# installs our package
uv pip install -e .
```

5.2 Next we have to downgrade `pytorch` and `pytorch lightning`, to the specific versions and commands we tested our code with:

```
uv pip install --reinstall torchmetrics====0.11.4
uv pip install monai==0.9.1
uv pip install --reinstall setuptools==69.5.1
uv pip install 'numpy==1.26.4'

# pytorch lightning 1.7.7
pip install pytorch_lightning==1.7.7

# pytorch 1.12.1 (as tested on our systems)
uv pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```


IMPORTANT: In order to use the equivariant Unet, i.e. the model that we found to work best, a fork of the `escnn` library instead of the original one has to be installed to avoid problems with pytorch lightning not being able to restore model checkpoints due to unpicklable objects.

Please check if `escnn` was already installed during the above process by trying

```
pip show escnn
```
If that works and a version of the library together with metadata is printed, you are good to go and you can skip the next part.

If you get a message like
```
WARNING: Package(s) not found: escnn"
```

we need to do the following:

```bash
# assuming you are still in the `src` sub-directory
cd ../..
git clone https://github.com/dmklee/escnn
cd escnn
# a specific commit hash that fixed the problem
git checkout e3c3c7944a83040706708c107071c9d1a6741c64
# install the state of the escnn library of that commit
pip install -e .
```

5.3 As a last (optional) step, we will create a jupyter kernel so the software stack from our created environment can directly
be used from within jupyter notebooks by executing

```
pip install ipykernel
python -m ipykernel install --user --name saxs --display-name "saxs"
```

This should allow you to select this kernel from the jupyter instance you might be running locally.

### Usage
For sending off batch jobs on a SLURM cluster (or execute scripts locally on your machine) that do model training or inference on patch data, make sure that
you have deactivated your conda environment (because the job scripts will activate it for you)

```
conda deactivate saxs
```
Your prompt should now be back to something like (the leading environment name in parenthesis should disappear)
```
yourname@yourcomputer:~/YOUR_PATH$
```


1. Model training on tiles on an HPC system
Training scripts can be found navigating to

```bash
cd src/scripts/02_model_training
# submit the job to a SLURM-based HPC system
sbatch run_train.sh
# or run the command locally by executing the line below
# ./run_train.sh
```

Before doing that, make sure to adjust the paths to the data accordingly within the script:
- SIGNAL_DIR: full path to a directory of your system that contains the tiles of signal-only data as individual tiff files.
- NOISE_DIR: full path to a subdirectory of your system that contains the tiles of noise-only data as individual tiff files.
- FILES_CSV_TRAIN_SIGNAL: path to a csv file containing the names of a subset of files within SIGNAL_DIR that will be used for training the model
- FILES_CSV_TRAIN_NOISE: path to a csv file containing the names of a subset of files within NOISE_DIR that will be used for training the model
- FILES_CSV_VAL_SIGNAL: path to a csv file containing the names of a subset of files within SIGNAL_DIR that will be used for  estimation of model performance during the training process (the model will not be trained on those!)
- FILES_CSV_VAL_NOISE: path to a csv file containing the names of a subset of files within NOISE_DIR that will be used for  estimation of model performance during the training process (the model will not be trained on those!)
- OUTPUT_DIR: a path where the trained model checkpoints will be written to (needed for making predictions later on)
- QUANTILE_LOW/QUANTILE_HIGH: the quantiles for intensity normalization of pixel intensities for the input tiles before being processed by the neural network

2. Model inference on tiles on an HPC system

Model prediction quality can be obtained for tiles in a batch job as well doing

```bash
# submit the job to a SLURM-based HPC system
sbatch run_inference_on_tiles.sh
# or run the command locally by executing the line below
# ./run_inference_on_tiles.sh
```
This will compute metrics (currently peak-signal-to-noise-ratio between estimated and ground truth clean signal, mean-absolute-error and mean-squared-error per pixel), but will not produce plots of predictions.

Before running this script, it is important to specify again the data via the above mentioned variables and, additionally the `TRAIN_OUTPUT_DIR` variable, as this needs to contain the trained model checkpoints which will be loaded to make predictions (usually this should match the `OUTPUT_DIR` variable set in the training script).

Also make sure that the QUANTILE_LOW/QUANTILE_HIGH parameters are set to the same values as used during training. Otherwise model prediction quality might be reduced.


3. Model inference on full images

A demo on how to apply a trained model on full-sized images is given in `src/scripts/03_inference/inference_demo.py`.


## Trained models

We provide access to our trained models through the following URL: https://doi.org/10.14278/rodare.4786

## Citation

If you find this code useful and beneficial in your work, please cite our work

TODO
```bash

```