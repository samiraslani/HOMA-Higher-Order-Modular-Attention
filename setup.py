from setuptools import find_packages, setup

setup(
    name="HOMA",
    version="0.1.0",
    description=(
        "Transformer with HOMA (Higher-Order Modular Attention) for protein sequence modelling"
    ),
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "tape_proteins>=0.5",
    ],
)
