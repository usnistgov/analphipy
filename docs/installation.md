# Installation

## Stable release

To install analphipy, run this command in your terminal:

```bash
pip install analphipy
```

or

```bash
conda install -c wpk-nist analphipy
```

This is the preferred method to install analphipy, as it will always install the
most recent stable release.

If you don't have [pip] installed, this [Python installation guide] can guide
you through the process.

## From sources

The sources for analphipy can be downloaded from the [Github repo].

You can either clone the public repository:

```bash
git clone git://github.com/usnistgov/analphipy.git
```

Once you have a copy of the source, you can install it with:

```bash
pip install .
```

To install dependencies with conda/mamba, use:

```bash
conda env create [-n {name}] -f environment.yaml
pip install [-e] --no-deps .
```

[github repo]: https://github.com/usnistgov/analphipy
[pip]: https://pip.pypa.io
[python installation guide]:
  http://docs.python-guide.org/en/latest/starting/installation/
