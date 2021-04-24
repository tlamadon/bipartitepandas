==========
Python API
==========

Overview
---------

The main BipartitePandas API is split into 8 classes, 3 of which are base classes. BipartitePandas is canonically imported using

  .. code-block:: python

    import bipartitepandas as bpd

Main classes
~~~~~~~~~~~~

* ``bipartitepandas.BipartiteLong``: Class for formatting bipartite networks in long form

* ``bipartitepandas.BipartiteLongCollapsed``: Class for formatting bipartite networks in collapsed long form (i.e. employment spells are collapsed into a single observation)

* ``bipartitepandas.BipartiteEventStudy``: Class for formatting bipartite networks in event study form

* ``bipartitepandas.BipartiteEventStudyCollapsed``: Class for formatting bipartite networks in collapsed event study form (i.e. employment spells are collapsed into a single observation)

* ``bipartitepandas.SimBipartite``: Class for simulating bipartite networks

Base classes
~~~~~~~~~~~~

* ``bipartitepandas.BipartiteBase``: Base class for BipartiteLongBase and BipartiteEventStudyBase. All methods are usable by any class that inherits from BipartiteBase.

* ``bipartitepandas.BipartiteLongBase``: Base class for BipartiteLong and BipartiteLongCollapsed. All methods are usable by any class that inherits from BipartiteLongBase.

* ``bipartitepandas.BipartiteEventStudyBase``: Base class for BipartiteEventStudy and BipartiteEventStudyCollapsed. All methods are usable by any class that inherits from BipartiteEventStudyBase.


Modules and Methods
-------------------

``bipartitepandas.BipartiteBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteBase
   ~bipartitepandas.BipartiteBase.clean_data
   ~bipartitepandas.BipartiteBase.cluster
   ~bipartitepandas.BipartiteBase.drop
   ~bipartitepandas.BipartiteBase.merge
   ~bipartitepandas.BipartiteBase.n_workers
   ~bipartitepandas.BipartiteBase.n_firms
   ~bipartitepandas.BipartiteBase.n_clusters
   ~bipartitepandas.BipartiteBase.original_ids
   ~bipartitepandas.BipartiteBase.rename
   ~bipartitepandas.BipartiteBase.summary

``bipartitepandas.BipartiteLongBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLongBase
   ~bipartitepandas.BipartiteLong.get_es

``bipartitepandas.BipartiteLong``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLong
   ~bipartitepandas.BipartiteLong.fill_periods
   ~bipartitepandas.BipartiteLong.get_collapsed_long

``bipartitepandas.BipartiteLongCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLongCollapsed
   ~bipartitepandas.BipartiteLongCollapsed.uncollapse

``bipartitepandas.BipartiteEventStudyBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudy
   ~bipartitepandas.BipartiteEventStudy.get_cs
   ~bipartitepandas.BipartiteEventStudy.get_long
   ~bipartitepandas.BipartiteEventStudy.unstack_es

``bipartitepandas.BipartiteEventStudy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudy

``bipartitepandas.BipartiteEventStudyCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudyCollapsed

``bipartitepandas.SimBipartite``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.SimBipartite
   ~bipartitepandas.SimBipartite.sim_network
