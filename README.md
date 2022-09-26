# analphipy

Utilities to perform metric analysis on fluid pair potentials. The main
features of `analphipy` as follows:

-   Pre-defined spherically symmetric potentials
-   Simple interface to extended to user defined pair potentials
-   Routines to calculate
    [Noro-Frenkel](https://en.wikipedia.org/wiki/Noro%E2%80%93Frenkel_law_of_corresponding_states)
    effective parameters.
-   Routines to calculate
    [Jensen-Shannon](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
    divergence

## Overview

`analphipy` is a python package to calculate metrics for classical
models for pair potentials. It provides a simple and extendable api for
pair potentials creation. Several routines to calculate metrics are
included in the package.


## Links

-   [Github](https://github.com/usnistgov/analphipy)
-   [Documentation](https://pages.nist.gov/analphipy/index.html)

## Examples

See [basic usage](docs/notebooks/usage.ipynb)

Or see the [full documentation](https://pages.nist.gov/analphipy/)


## Installation

``` bash
pip install analphipy
```

or

``` bash
conda install -c wpk-nist analphipy
```

## Status

This package is actively used by the author.  Please feel free to create a pull request for wanted features and suggestions!

## License

This is free software. See [LICENSE](LICENSE)

## Contact

The author can be reached at <wpk@nist.gov>.

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[wpk-nist-gov/cookiecutter-pypackage](https://github.com/wpk-nist-gov/cookiecutter-pypackage)
Project template forked from
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
