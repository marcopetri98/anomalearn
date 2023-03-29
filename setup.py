from setuptools import setup, find_packages


def long_description():
    """Gets the long description.
    
    Returns
    -------
    long_description
        The long description.
    """
    with open("README.md", "r") as f:
        content = f.read()
        
    return content


def setup_package():
    setup(
        name="anomalearn",
        version="0.0.1a1",
        description="A library aiding the development of time series anomaly detection methods using modular components.",
        long_description=long_description(),
        long_description_content_type="text/markdown",
        author="Marco Petri",
        author_email="marco.petri@mail.polimi.it",
        license="Mozilla Public License 2.0",
        packages=find_packages(exclude=["generator*", "tests*"]),
        keywords="time series anomaly detection development machine learning",
        python_requires=">=3.10",
        install_requires=[
            "colorama>=0.4.4",
            "matplotlib>=3.5.1",
            "numba>=0.56.4",
            "numpy>=1.21.5",
            "pandas>=1.4.1",
            "scikit-learn>=1.0.2",
            "scikit-optimize>=0.9.0",
            "scipy>=1.7.3",
            "seaborn>=0.11.2",
            "statsmodels>=0.13.0",
            "tensorflow>=2.11.0",
            "tqdm>=4.64.1",
            "urllib3>=1.26.9"
            ],
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries"
            ],
        url="https://github.com/marcopetri98/2021-2022-thesis-petri",
        zip_safe=False
        )


if __name__ == "__main__":
    setup_package()
