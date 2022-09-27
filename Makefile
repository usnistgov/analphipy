.PHONY: clean clean-test clean-pyc clean-build help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -rf docs/_build
	rm -rf conda-build/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache


################################################################################
# utilities
################################################################################
.PHONY: lint pre-commit-init pre-commit-run pre-commit-run-all init

lint: ## check style with flake8
	flake8 analphipy tests

pre-commit-init: ## install pre-commit
	pre-commit install

pre-commit-run: ## run pre-commit
	pre-commit run

pre-commit-run-all: ## run pre-commit on all files
	pre-commit run --all-files

.git: ## init git
	git init


init: .git pre-commit-init ## run git-init pre-commit


################################################################################
# virtual env
################################################################################
.PHONY: mamba-env mamba-dev mamba-env-update mamba-dev-update activate

environment-dev.yaml: environment.yaml environment-tools.yaml
	conda-merge environment.yaml environment-tools.yaml > environment-dev.yaml

mamba-env: environment.yaml
	mamba env create -f environment.yaml

mamba-dev: environment-dev.yaml
	mamba env create -f environment-dev.yaml

mamba-env-update: environment.yaml
	mamba env update -f environment.yml

mamba-dev-update: environment-dev.yaml
	mamba env update -f environment-dev.yml

activate: ## activate base env
	conda activate {{ cookiecutter.project_slug }}-env



################################################################################
# my convenience functions
################################################################################
.PHONY: user-venv user-autoenv-zsh user-all
user-venv: ## create .venv file with name of conda env
	echo analphipy-env > .venv

user-autoenv-zsh: ## create .autoenv.zsh files
	echo conda activate $$(cat .venv) > .autoenv.zsh
	echo conda deactivate > .autoenv_leave.zsh

user-all: user-venv user-autoenv-zsh ## runs user scripts


################################################################################
# Testing
################################################################################
.PHONY: test test-all coverage
test: ## run tests quickly with the default Python
	pytest -x -v

test-all: ## run tests on every Python version with tox
	tox -- -x -v

coverage: ## check code coverage quickly with the default Python
	coverage run --source analphipy -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html


################################################################################
# versioning
################################################################################
.PHONY: version-scm version-import version
version-scm: ## check version of package
	python -m setuptools_scm

version-import: ## check version from python import
	python -c 'import analphipy; print(analphipy.__version__)'

version: version-scm version-import

################################################################################
# Docs
################################################################################
.PHONY: docs serverdocs
docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/analphipy.rst
	rm -f docs/modules.rst
	rm -fr docs/generated
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

doc-spelling:
	sphinx-build -b spelling docs docs/_build


################################################################################
# distribution
###############################################################################
dist: ## builds source and wheel package (run clean?)
	python -m build
	ls -l dist


.PHONY: release release-test conda-dist
release: dist ## package and upload a release
	twine upload dist/*

release-test: dist ## package and upload to test
	twine upload --repository testpypi dist/*


conda-greyskull:
	mkdir -p conda_dist && \
	cd conda_dist && \
	grayskull pypi analphipy && \
	cat analphipy/meta.yaml

conda-build:
	cd conda-dist && \
	conda-build . && \
	echo 'upload now'

conda-dist: conda-greyskull conda-build


################################################################################
# installation
################################################################################
.PHONY: install install-dev
install: ## install the package to the active Python's site-packages (run clean?)
	python setup.py install

install-dev: ## install development version (run clean?)
	pip install -e . --no-deps
