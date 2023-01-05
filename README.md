# Description
MPH, or Many-Particle Holstein is essentially just a Python translation of the exciton1D 
code written by Dr. Nicholas Hestand (https://github.com/nicholashestand/exciton1d). The code
constructs and diagonalizes the Frenkel-Holstein Hamiltonian for a molecular aggregate
in the basis of 1- and 2-particle excitonic states, and uses the resulting eigenvalues
and eigenvectors to compute the absorption spectrum. In the future, I plan on adding
a few things, including a PDF showing derivations for all matrix elements, 3-particle and higher
terms, and higher-order spectroscopy simulations, such as pump-prope and 2D electronic spectroscopy.

# Installation
Create a conda environment with Numpy, Matplotlib, and Numba installed. Clone this repository and run
make. The given makefile will install the package locally and it can be used as shown in the examples.
For convenience, a yml file is added so that all you need to do is

```conda env create -f environment.yml```

The environment name by default is ```mph_env```. Then, activate the environment with

```conda activate mph_env```,

and run ```make```.

