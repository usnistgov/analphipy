.. currentmodule:: analphipy

API Reference
=============

Defined potentials
------------------

These potentials have defined values for :math:`\phi(r)` as well as exact values for :math:`r_{\rm min}`, if appropriate.


.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   Phi_lj
   Phi_nm
   Phi_yk
   Phi_sw
   Phi_hs

.. autosummary::
   :toctree: generated/

   factory_phi




Base classes for potentials
---------------------------

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   PhiBase
   PhiAnalytic
   PhiGeneric
   PhiCut
   PhiLFS



Analysis
--------

``norofrenkel`` module
^^^^^^^^^^^^^^^^^^^^^^


.. autosummary::
   :toctree: generated/
   :template: custom-class.rst


   NoroFrenkelPair


.. currentmodule:: analphipy.norofrenkel

.. autosummary::
   :toctree: generated/

   sig_nf
   sig_nf_dbeta
   lam_nf
   lam_nf_dbeta


``measures`` module
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: analphipy.measures


.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   Measures

.. autosummary::
   :toctree: generated/

   secondvirial
   secondvirial_dbeta
   secondvirial_sw
   diverg_kl_cont
   diverg_js_cont


``utils`` module
^^^^^^^^^^^^^^^^

.. currentmodule:: analphipy.utils

.. autosummary::
   :toctree: generated/

   quad_segments
   minimize_phi
