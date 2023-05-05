<!-- markdownlint-disable MD041 -->

[![Repo][repo-badge]][repo-link] [![Docs][docs-badge]][docs-link]
[![PyPI license][license-badge]][license-link]
[![PyPI version][pypi-badge]][pypi-link]
[![Conda (channel only)][conda-badge]][conda-link]
[![Code style: black][black-badge]][black-link]

<!-- For more badges, see
  https://shields.io/category/other
  https://naereen.github.io/badges/
  [pypi-badge]: https://badge.fury.io/py/analphipy
-->

[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/psf/black
[pypi-badge]: https://img.shields.io/pypi/v/analphipy
[pypi-link]: https://pypi.org/project/analphipy
[docs-badge]: https://img.shields.io/badge/docs-sphinx-informational
[docs-link]: https://pages.nist.gov/analphipy/
[repo-badge]: https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff
[repo-link]: https://github.com/usnistgov/analphipy
[conda-badge]: https://img.shields.io/conda/v/wpk-nist/analphipy
[conda-link]: https://anaconda.org/wpk-nist/analphipy
[license-badge]: https://img.shields.io/pypi/l/cmomy?color=informational
[license-link]: https://github.com/usnistgov/analphipy/blob/main/LICENSE

<!-- other links -->

[jensen-shannon]:
  https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
[noro-frenkel]:
  https://en.wikipedia.org/wiki/Noro%E2%80%93Frenkel_law_of_corresponding_states

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

## Quick start

Use one of the following to install `analphipy`

```bash
pip install analphipy
```

or

```bash
conda install -c wpk-nist analphipy
```

## Example usage

```python
# Create a Lennard-Jones potential
>>> import analphipy
>>> p = analphipy.potential.LennardJones(sig=1.0, eps=1.0)

# Get a Noro-Frenekl analysis object
>>> n = p.to_nf()

# Get effective parameters at inverse temperature beta
>>> n.sig(beta=1.0)
1.01560...

>>> n.eps(beta=1.0)
-1.0

>>> n.lam(beta=1.0)
1.44097...

```

<!-- end-docs -->

## Documentation

See the [documentation][docs-link] for a look at `analphipy` in action.

## License

This is free software. See [LICENSE][license-link].

## Contact

The author can be reached at wpk@nist.gov.

## Credits

This package was created with [Cookiecutter] and the
[wpk-nist-gov/cookiecutter-pypackage] Project template forked from
[audreyr/cookiecutter-pypackage].

[audreyr/cookiecutter-pypackage]:
  https://github.com/audreyr/cookiecutter-pypackage
[cookiecutter]: https://github.com/audreyr/cookiecutter
[wpk-nist-gov/cookiecutter-pypackage]:
  https://github.com/wpk-nist-gov/cookiecutter-pypackage
