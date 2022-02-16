==========
Python API
==========

Overview
---------

The main BipartitePandas API is split into eight classes, three of which are base classes and one of which is for simulating bipartite data. It also has two modules for clustering: one for computing measures and one for grouping on measures. BipartitePandas is canonically imported using

  .. code-block:: python

    import bipartitepandas as bpd

Main classes
~~~~~~~~~~~~

* ``bipartitepandas.BipartiteDataFrame``: Class to easily construct bipartite dataframes without explicitly specifying a format

* ``bipartitepandas.BipartiteLong``: Class for formatting bipartite data in long form

* ``bipartitepandas.BipartiteLongCollapsed``: Class for formatting bipartite data in collapsed long form (i.e. employment spells are collapsed into a single observation)

* ``bipartitepandas.BipartiteEventStudy``: Class for formatting bipartite data in event study form

* ``bipartitepandas.BipartiteEventStudyCollapsed``: Class for formatting bipartite data in collapsed event study form (i.e. employment spells are collapsed into a single observation)

* ``bipartitepandas.SimBipartite``: Class for simulating bipartite data

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

``bipartitepandas.BipartiteDataFrame``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteDataFrame

``bipartitepandas.BipartiteBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteBase
   ~bipartitepandas.BipartiteBase.add_column
   ~bipartitepandas.BipartiteBase.cluster
   ~bipartitepandas.BipartiteBase.copy
   ~bipartitepandas.BipartiteBase.diagnostic
   ~bipartitepandas.BipartiteBase.drop
   ~bipartitepandas.BipartiteBase.drop_rows
   ~bipartitepandas.BipartiteBase.get_column_properties
   ~bipartitepandas.BipartiteBase.log
   ~bipartitepandas.BipartiteBase.log_on
   ~bipartitepandas.BipartiteBase.merge
   ~bipartitepandas.BipartiteBase.min_movers_firms
   ~bipartitepandas.BipartiteBase.n_clusters
   ~bipartitepandas.BipartiteBase.n_firms
   ~bipartitepandas.BipartiteBase.n_unique_ids
   ~bipartitepandas.BipartiteBase.n_workers
   ~bipartitepandas.BipartiteBase.original_ids
   ~bipartitepandas.BipartiteBase.print_column_properties
   ~bipartitepandas.BipartiteBase.rename
   ~bipartitepandas.BipartiteBase.set_column_properties
   ~bipartitepandas.BipartiteBase.sort_cols
   ~bipartitepandas.BipartiteBase.sort_rows
   ~bipartitepandas.BipartiteBase.summary
   ~bipartitepandas.BipartiteBase.unique_ids

``bipartitepandas.BipartiteLongBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLongBase
   ~bipartitepandas.BipartiteLongBase.clean
   ~bipartitepandas.BipartiteLongBase.construct_artificial_time
   ~bipartitepandas.BipartiteLongBase.drop_ids
   ~bipartitepandas.BipartiteLongBase.gen_m
   ~bipartitepandas.BipartiteLongBase.keep_ids
   ~bipartitepandas.BipartiteLongBase.keep_rows
   ~bipartitepandas.BipartiteLongBase.min_movers_frame
   ~bipartitepandas.BipartiteLongBase.min_moves_firms
   ~bipartitepandas.BipartiteLongBase.min_moves_frame
   ~bipartitepandas.BipartiteLongBase.min_obs_firms
   ~bipartitepandas.BipartiteLongBase.min_obs_frame
   ~bipartitepandas.BipartiteLongBase.min_workers_firms
   ~bipartitepandas.BipartiteLongBase.min_workers_frame
   ~bipartitepandas.BipartiteLongBase.to_eventstudy

``bipartitepandas.BipartiteLong``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLong
   ~bipartitepandas.BipartiteLong.collapse
   ~bipartitepandas.BipartiteLong.fill_periods
   ~bipartitepandas.BipartiteLong.get_extended_eventstudy
   ~bipartitepandas.BipartiteLong.get_worker_m
   ~bipartitepandas.BipartiteLong.plot_extended_eventstudy

``bipartitepandas.BipartiteLongCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLongCollapsed
   ~bipartitepandas.BipartiteLongCollapsed.get_worker_m
   ~bipartitepandas.BipartiteLongCollapsed.recollapse
   ~bipartitepandas.BipartiteLongCollapsed.uncollapse

``bipartitepandas.BipartiteEventStudyBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudyBase
   ~bipartitepandas.BipartiteEventStudyBase.clean
   ~bipartitepandas.BipartiteEventStudyBase.construct_artificial_time
   ~bipartitepandas.BipartiteEventStudyBase.diagnostic
   ~bipartitepandas.BipartiteEventStudyBase.drop_ids
   ~bipartitepandas.BipartiteEventStudyBase.gen_m
   ~bipartitepandas.BipartiteEventStudyBase.get_cs
   ~bipartitepandas.BipartiteEventStudyBase.keep_ids
   ~bipartitepandas.BipartiteEventStudyBase.keep_rows
   ~bipartitepandas.BipartiteEventStudyBase.min_movers_frame
   ~bipartitepandas.BipartiteEventStudyBase.min_moves_firms
   ~bipartitepandas.BipartiteEventStudyBase.min_moves_frame
   ~bipartitepandas.BipartiteEventStudyBase.min_obs_firms
   ~bipartitepandas.BipartiteEventStudyBase.min_obs_frame
   ~bipartitepandas.BipartiteEventStudyBase.min_workers_firms
   ~bipartitepandas.BipartiteEventStudyBase.min_workers_frame
   ~bipartitepandas.BipartiteEventStudyBase.to_long

``bipartitepandas.BipartiteEventStudy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudy
   ~bipartitepandas.BipartiteEventStudy.collapse
   ~bipartitepandas.BipartiteEventStudy.get_worker_m

``bipartitepandas.BipartiteEventStudyCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudyCollapsed
   ~bipartitepandas.BipartiteEventStudyCollapsed.get_worker_m
   ~bipartitepandas.BipartiteEventStudyCollapsed.uncollapse

``bipartitepandas.SimBipartite``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.SimBipartite
   ~bipartitepandas.SimBipartite.simulate

Modules and Methods
-------------------

``bipartitepandas.measures``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.measures.CDFs
   ~bipartitepandas.measures.Moments

``bipartitepandas.grouping``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.grouping.KMeans
   ~bipartitepandas.grouping.Quantiles
