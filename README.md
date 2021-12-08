# MOT
Approximative algorithms for computing sparse Wasserstein-2 multi-marginal optimal transport plans and corresponding free support barycenters.
Using the `emd` OT solver from the [Python Optimal Transport (POT)](https://pythonot.github.io/index.html) package, which is a wrapper of [this network simplex solver](https://github.com/nbonneel/network_simplex), which, in turn, is based on an implementation in the [LEMON](http://lemon.cs.elte.hu/pub/doc/latest-svn/index.html) C++ library.

Johannes von Lindheim, 2021

## Installation
1. Download the code or clone the Github repository with
```
git clone https://github.com/jvlindheim/mot.git
```
2. For the code in `mot.py`, there is the following dependencies: `numpy`, `matplotlib.pyplot`, the `cdist` function from `scipy.spatial.distance` and the `emd` function from the [POT](https://pythonot.github.io/index.html) library. You can install them e.g. using pip via
```
pip install --user numpy scipy matplotlib POT
```
If you want to run the demo notebook, you will also need to have [Jupyter Notebook or JupyterLab](https://jupyter.org/install) installed.
