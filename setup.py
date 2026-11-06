from setuptools import setup, find_packages

setup(
    name='failure_prob',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.10,<3.11',
    install_requires=[
        # Mirrors the README "Install other required packages" list, plus torch.
        # For GPU builds, install torch from the CUDA index first (see README).
        'torch',
        'torchvision',
        'pandas',
        'scipy',
        'pyyaml',
        'tqdm',
        'imageio[ffmpeg]',
        'hydra-core',
        'omegaconf',
        'scikit-learn',
        'opencv_python',
        'einops',
        'wandb',
        'plotly',
        'matplotlib',
        'natsort',
        'flask',
        'ml_dtypes',
        'pytest',
    ],
)
