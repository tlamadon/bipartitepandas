==========
Python API
==========

Overview
---------

The main BipartitePandas API is split into five classes. BipartitePandas is canonically imported using

  .. code-block:: python

    import bipartitepandas as bpd

* ``bipartitepandas.BipartiteLong``: Class for formatting bipartite networks in long form

* ``bipartitepandas.BipartiteLongCollapsed``: Class for formatting bipartite networks in collapsed long form (i.e. employment spells are collapsed into a single observation)

* ``bipartitepandas.BipartiteEventStudy``: Class for formatting bipartite networks in event study form

* ``bipartitepandas.BipartiteEventStudyCollapsed``: Class for formatting bipartite networks in collapsed event study form (i.e. employment spells are collapsed into a single observation)

* ``bipartitepandas.SimBipartite``: Class for simulating bipartite networks

Modules and Methods
-------------------

``bipartitepandas.BipartiteLong``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLong
   ~bipartitepandas.BipartiteLong.clean_data
   ~bipartitepandas.BipartiteLong.cluster
   ~bipartitepandas.BipartiteLong.drop
   ~bipartitepandas.BipartiteLong.fill_periods
   ~bipartitepandas.BipartiteLong.get_collapsed_long
   ~bipartitepandas.BipartiteLong.get_es
   ~bipartitepandas.BipartiteLong.merge
   ~bipartitepandas.BipartiteLong.n_workers
   ~bipartitepandas.BipartiteLong.n_firms
   ~bipartitepandas.BipartiteLong.n_clusters
   ~bipartitepandas.BipartiteLong.original_ids
   ~bipartitepandas.BipartiteLong.rename
   ~bipartitepandas.BipartiteLong.summary

``bipartitepandas.BipartiteLongCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLongCollapsed
   ~bipartitepandas.BipartiteLongCollapsed.clean_data
   ~bipartitepandas.BipartiteLongCollapsed.cluster
   ~bipartitepandas.BipartiteLongCollapsed.drop
   ~bipartitepandas.BipartiteLongCollapsed.get_es
   ~bipartitepandas.BipartiteLongCollapsed.merge
   ~bipartitepandas.BipartiteLongCollapsed.n_workers
   ~bipartitepandas.BipartiteLongCollapsed.n_firms
   ~bipartitepandas.BipartiteLongCollapsed.n_clusters
   ~bipartitepandas.BipartiteLongCollapsed.original_ids
   ~bipartitepandas.BipartiteLongCollapsed.rename
   ~bipartitepandas.BipartiteLongCollapsed.summary
   ~bipartitepandas.BipartiteLongCollapsed.uncollapse

``bipartitepandas.BipartiteEventStudy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudy
   ~bipartitepandas.BipartiteEventStudy.clean_data
   ~bipartitepandas.BipartiteEventStudy.cluster
   ~bipartitepandas.BipartiteEventStudy.drop
   ~bipartitepandas.BipartiteEventStudy.get_cs
   ~bipartitepandas.BipartiteEventStudy.get_long
   ~bipartitepandas.BipartiteEventStudy.merge
   ~bipartitepandas.BipartiteEventStudy.n_workers
   ~bipartitepandas.BipartiteEventStudy.n_firms
   ~bipartitepandas.BipartiteEventStudy.n_clusters
   ~bipartitepandas.BipartiteEventStudy.original_ids
   ~bipartitepandas.BipartiteEventStudy.rename
   ~bipartitepandas.BipartiteEventStudy.summary
   ~bipartitepandas.BipartiteEventStudy.unstack_es

``bipartitepandas.BipartiteEventStudyCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudyCollapsed
   ~bipartitepandas.BipartiteEventStudyCollapsed.clean_data
   ~bipartitepandas.BipartiteEventStudyCollapsed.cluster
   ~bipartitepandas.BipartiteEventStudyCollapsed.drop
   ~bipartitepandas.BipartiteEventStudyCollapsed.get_cs
   ~bipartitepandas.BipartiteEventStudyCollapsed.get_collapsed_long
   ~bipartitepandas.BipartiteEventStudyCollapsed.merge
   ~bipartitepandas.BipartiteEventStudyCollapsed.n_workers
   ~bipartitepandas.BipartiteEventStudyCollapsed.n_firms
   ~bipartitepandas.BipartiteEventStudyCollapsed.n_clusters
   ~bipartitepandas.BipartiteEventStudyCollapsed.original_ids
   ~bipartitepandas.BipartiteEventStudyCollapsed.rename
   ~bipartitepandas.BipartiteEventStudyCollapsed.summary
   ~bipartitepandas.BipartiteEventStudyCollapsed.unstack_es

``bipartitepandas.SimBipartite``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.SimBipartite
   ~bipartitepandas.SimBipartite.sim_network()
