==========
Python API
==========

Overview
---------

The main BipartitePandas API is split into 8 classes, 3 of which are base classes. It also has 2 modules for clustering: one for computing measures and one for grouping on measures. BipartitePandas is canonically imported using

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

Clustering modules
~~~~~~~~~~~~~~~~~~

* ``bipartitepandas.measures``: Module for computing measures

* ``bipartitepandas.grouping``: Module for grouping on measures

Classes and Methods
-------------------

``bipartitepandas.BipartiteBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteBase
   ~bipartitepandas.BipartiteBase.clean_data
   ~bipartitepandas.BipartiteBase.cluster
   ~bipartitepandas.BipartiteBase.copy
   ~bipartitepandas.BipartiteBase.drop
   ~bipartitepandas.BipartiteBase.gen_m
   ~bipartitepandas.BipartiteBase.merge
   ~bipartitepandas.BipartiteBase.n_clusters
   ~bipartitepandas.BipartiteBase.n_firms
   ~bipartitepandas.BipartiteBase.n_workers
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
   ~bipartitepandas.BipartiteLong.get_es_extended
   ~bipartitepandas.BipartiteLong.plot_es_extended

``bipartitepandas.BipartiteLongCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLongCollapsed
   ~bipartitepandas.BipartiteLongCollapsed.uncollapse

``bipartitepandas.BipartiteEventStudyBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudyBase
   ~bipartitepandas.BipartiteEventStudyBase.clean_data
   ~bipartitepandas.BipartiteEventStudyBase.get_cs
   ~bipartitepandas.BipartiteEventStudyBase.get_long
   ~bipartitepandas.BipartiteEventStudyBase.unstack_es

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

Modules and Methods
-------------------

``bipartitepandas.measures``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.measures.cdfs
   ~bipartitepandas.measures.moments

``bipartitepandas.grouping``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.grouping.kmeans
   ~bipartitepandas.grouping.quantiles
