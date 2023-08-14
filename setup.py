#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="pysparkplug",
    version="0.1.9.0",
    description="A package for estimating heterogeneous probability density functions.",
    author="Grant Boquet",
    author_email="grant.boquet@gmail.com",
    url="N/A",
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
    install_requires=[
        "scipy",
        "matplotlib",
        "numpy",
        "numba",
        "mpmath",
        "pandas",
        "pyspark",
        "tbb",
    ],

)
