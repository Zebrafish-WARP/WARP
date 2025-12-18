import setuptools
from setuptools import setup

install_deps = [
    'numpy>=1.20.0',
    'scipy==1.10.1',
    'ipykernel>=6.15.0',
    'matplotlib>=3.7.0',
    'seaborn>=0.12.0',
    'notebook>=6.5.0',        # Jupyter Notebook
    'pynrrd>=1.0.0',          # NRRD
    'scikit-learn>=1.2.0',    # sklearn
    'distinctipy>=1.1.0',
    'rastermap>=0.6.0',
    'tqdm-joblib>=0.0.3',
    'statsmodels>=0.14.0',
    'fishspot'
]

# ipywidgets
# python-igraph leidenalg

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="WARP",
    license="BSD",
    author="Ahrens Lab",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/",
    setup_requires=[
        'pytest-runner',
        'setuptools_scm',
    ],
    packages=setuptools.find_packages(),
    use_scm_version=True,
    install_requires=install_deps,
    tests_require=[
        'pytest'
    ],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
)