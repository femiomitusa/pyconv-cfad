from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="radar_processing",
    version="0.1.0",
    author="Oluwafemi Omitusa",
    description="A Python package for processing weather radar data and tracking storm cells",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oluwafemiomitusa/radar-cell-processing",
    packages=find_packages(include=['radar_processing', 'radar_processing.*']),
    package_data={
        'radar_processing': ['myutils/*.py'],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "arm-pyart>=1.13.0",
        "pandas>=1.3.0",
        "xarray>=0.19.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.19.0",
        "tqdm>=4.65.0",
        "zarr>=2.16.0",
    ],
)
