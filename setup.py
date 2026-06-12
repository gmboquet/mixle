#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="pysparkplug",
    version="0.2.0.0",
    description="A package for estimating heterogeneous probability density functions.",
    author="Grant Boquet",
    author_email="grant.boquet@gmail.com",
    url="https://github.com/gmboquet/pysparkplug",
    packages=find_packages(),
    long_description="""\
    A package for estimating heterogeneous probability density functions.
    """,
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
    ],
    keywords="machine learning density estimation statistics heterogeneous data",
    license="MIT",
    # the base install covers all distributions and local (numpy) estimation;
    # acceleration, distribution, and embedding back-ends are opt-in extras
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "mpmath",
        "networkx",
        "tqdm",
    ],
    extras_require={
        "numba": ["numba", "tbb"],
        "spark": ["pyspark"],
        "torch": ["torch"],
        "mpi": ["mpi4py"],
        "umap": ["umap-learn"],
        "test": ["pytest>=8"],
        "all": ["numba", "tbb", "pyspark", "torch", "mpi4py", "umap-learn"],
    },
)
