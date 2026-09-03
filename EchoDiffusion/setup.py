from setuptools import setup, find_packages

setup(
    name="echodiffusion",
    version="0.1.0",
    description="Audio-guided diffusion policy for trajectory generation toward a sound source",
    packages=find_packages(include=["echodiffusion", "echodiffusion.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "image": ["timm>=0.9.0", "einops>=0.6.1", "opencv-python>=4.8.0"],
        "logging": ["comet-ml>=3.35.0"],
    },
)
