```{highlight} shell
```

# Installation

## Stable release

To install `analphipy`, run one of the following command:

```console
# from pip
$ pip install analphipy

# from conda/mamba
$ conda install -c wpk-nist analphipy
```

## From sources

The sources for `analphipy` can be downloaded from the [Github repo].

You can either clone the public repository:

```console
$ git clone {repo}
```

Once you have a copy of the source, you can install it with:

```console
# You may want a seperate virtual environment.  You can create a conda env with
$ conda env create -n {env-name} -f environment-dev.yml
$ conda activate {env-name}

# install editable package
$ pip install -e . --no-deps
```

[github repo]: https://github.com/usnistgov/analphipy
