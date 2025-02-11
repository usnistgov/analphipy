<!-- markdownlint-disable MD041 -->

<!-- prettier-ignore-start -->
[![Repo][repo-badge]][repo-link]
[![Docs][docs-badge]][docs-link]
[![PyPI license][license-badge]][license-link]
[![PyPI version][pypi-badge]][pypi-link]
[![Conda (channel only)][conda-badge]][conda-link]
[![Code style: ruff][ruff-badge]][ruff-link]
[![uv][uv-badge]][uv-link]

<!-- For more badges, see
  https://shields.io/category/other
  https://naereen.github.io/badges/
  [pypi-badge]: https://badge.fury.io/py/analphipy
-->

[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-link]: https://github.com/astral-sh/ruff
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[uv-link]: https://github.com/astral-sh/uv
[pypi-badge]: https://img.shields.io/pypi/v/analphipy
[pypi-link]: https://pypi.org/project/analphipy
[docs-badge]: https://img.shields.io/badge/docs-sphinx-informational
[docs-link]: https://pages.nist.gov/analphipy/
[repo-badge]: https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff
[repo-link]: https://github.com/usnistgov/analphipy
[conda-badge]: https://img.shields.io/conda/v/conda-forge/analphipy
[conda-link]: https://anaconda.org/conda-forge/analphipy
[license-badge]: https://img.shields.io/pypi/l/analphipy?color=informational
[license-link]: https://github.com/usnistgov/analphipy/blob/main/LICENSE

<!-- other links -->

[jensen-shannon]: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
[noro-frenkel]: https://en.wikipedia.org/wiki/Noro%E2%80%93Frenkel_law_of_corresponding_states

<!-- prettier-ignore-end -->

# `analphipy`

Utilities to perform metric analysis on fluid pair potentials. The main features
of `analphipy` as follows:

## Overview

`analphipy` is a python package to calculate metrics for classical models for
pair potentials. It provides a simple and extendable api for pair potentials
creation. Several routines to calculate metrics are included in the package.

## Features

- Pre-defined spherically symmetric potentials
- Simple interface to extended to user defined pair potentials
- Routines to calculate [Noro-Frenkel] effective parameters.
- Routines to calculate [Jensen-Shannon] divergence

## Status

This package is actively used by the author. Please feel free to create a pull
request for wanted features and suggestions!

## Example usage

```pycon
# Create a Lennard-Jones potential
>>> import analphipy
>>> p = analphipy.potential.LennardJones(sig=1.0, eps=1.0)

# Get a Noro-Frenekl analysis object
>>> n = p.to_nf()

# Get effective parameters at inverse temperature beta
>>> print(n.sig(beta=1.0))
1.01560...

>>> print(n.eps(beta=1.0))
-1.0

>>> print(n.lam(beta=1.0))
1.44097...

```

<!-- end-docs -->

## Installation

<!-- start-installation -->

Use one of the following to install `analphipy`

```bash
pip install analphipy
```

or

```bash
conda install -c conda-forge analphipy
```

<!-- end-installation -->

## Documentation

See the [documentation][docs-link] for a look at `analphipy` in action.

## License

This is free software. See [LICENSE][license-link].

## Contact

The author can be reached at <wpk@nist.gov>.

## Credits

This package was created using
[Cookiecutter](https://github.com/audreyr/cookiecutter) with the
[usnistgov/cookiecutter-nist-python](https://github.com/usnistgov/cookiecutter-nist-python)
template.
