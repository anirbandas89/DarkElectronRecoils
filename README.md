# Electron recoils in dark matter detectors (work in progress)

[//]:[![DOI](https://zenodo.org/badge/156694427.svg)](https://zenodo.org/badge/latestdoi/156694427)
[//]:[![arXiv](https://img.shields.io/badge/arXiv-1909.04684-B31B1B.svg)](https://arxiv.org/abs/1909.04684)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)


Please contact me at ciaran.aj.ohare@gmail.com if you want to complain about why something doesn't work.

## Contents

The code, plots, datas, etc. are sorted as follows:

* `data/` - various bits of required data
* `erec/` - main body of the code
* `notebooks/` - Ipython notebooks for plotting and going through the code
* `plots/` - plots get put here

## Requirements

The code is all written in python3 and makes substantial use of the standard numpy, matplotlib, scipy etc. There are several additonal libraries that you may need to investigate depending on your installation:

* [`astropy`](https://www.astropy.org/), used for various things
* [`cmocean`](https://matplotlib.org/cmocean/), aesthetic colormaps
* [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/), used to make Mollweide skymaps
* [`healpy`](https://healpy.readthedocs.io/en/latest/), used for speeding up angular integrals


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
