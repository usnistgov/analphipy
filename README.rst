=========
analphipy
=========

Utilities to perform metric analysis on fluid pair potentials.  The main features of ``analphipy`` as follows:

* Pre-defined spherically symmetric potentials
* Simple interface to extended to user defined pair potentials
* Routines to calculate `Noro-Frenkel`_ effective parameters.
* Routines to calculate `Jensen-Shannon`_ divergence

Overview
--------

``analphipy`` is a python package to calculate metrics for classical models for pair potentials.  It provides a simple and extendable api for pair potentials creation.  Several routines to calculate metrics are included in the package.

Links
-----

* `Github <https://github.com/usnistgov/analphipy>`__
* `Documentation <https://pages.nist.gov/analphipy/index.html>`__


Examples
--------

See `basic usage <docs/notebooks/usage.ipynb>`_


Or see the `full documentation <https://pages.nist.gov/analphipy/>`_


License
-------

This is free software.  See `LICENSE <LICENSE>`_


Contact
-------

The author can be reached at wpk@nist.gov.


Credits
-------

This package was created with Cookiecutter_ and the `wpk-nist-gov/cookiecutter-pypackage`_ Project template forked from `audreyr/cookiecutter-pypackage`_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`wpk-nist-gov/cookiecutter-pypackage`: https://github.com/wpk-nist-gov/cookiecutter-pypackage
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`Noro-Frenkel`: https://en.wikipedia.org/wiki/Noro%E2%80%93Frenkel_law_of_corresponding_states
.. _`Jensen-Shannon`: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
