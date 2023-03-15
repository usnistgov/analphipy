# API Reference

## Defined potentials

These potentials have defined values for $\phi(r)$ as well as exact values for $r_{\rm min}$, if appropriate.

```{eval-rst}
.. currentmodule:: analphipy
.. autosummary::
   :toctree: generated/

   Phi_lj
   Phi_nm
   Phi_yk
   Phi_sw
   Phi_hs
   factory_phi
```

## Base classes for potentials

```{eval-rst}
.. autosummary::
   :toctree: generated/

   PhiBase
   PhiAnalytic
   PhiGeneric
   PhiCut
   PhiLFS
```

## Analysis

### `norofrenkel` module

```{eval-rst}
.. autosummary::
   :toctree: generated/

   NoroFrenkelPair

```

```{eval-rst}
.. currentmodule:: analphipy.norofrenkel

.. autosummary::
   :toctree: generated/

   sig_nf
   sig_nf_dbeta
   lam_nf
   lam_nf_dbeta

```

### `measures` module

```{eval-rst}
.. currentmodule:: analphipy.measures

.. autosummary::
   :toctree: generated/

   Measures
   secondvirial
   secondvirial_dbeta
   secondvirial_sw
   diverg_kl_cont
   diverg_js_cont

```

### `utils` module

```{eval-rst}
.. currentmodule:: analphipy.utils

.. autosummary::
   :toctree: generated/

   quad_segments
   minimize_phi
```
