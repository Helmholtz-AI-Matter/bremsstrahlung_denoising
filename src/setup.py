from setuptools import find_packages, setup

setup(
    name='bremsstrahlung_denoising',
    packages=find_packages(),
    version='0.1.0',
    description='Denoising of SAXS data',
    author='Sebastian Starke, Michal Smid, Peter Steinbach',
    license='MIT',
    install_requires=[
        'matplotlib',
        'monai==0.9.1',
        'numpy==1.26.4',
        'pandas',
        'pytorch_lightning==1.7.7',
        'setuptools==69.5.1',
        'scikit_learn',
        'scikit_image',
        'tifffile',
        'torch==1.12.1',
        'torchmetrics==0.11.4',
        'tqdm',
        'escnn @ git+https://github.com/dmklee/escnn.git@e3c3c7944a83040706708c107071c9d1a6741c64'
    ]
)
