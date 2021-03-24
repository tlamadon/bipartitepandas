==========
Python API
==========

Overview
---------

The main bipartitepandas API is split into four classes. bipartitepandas is canonically imported using

  .. code-block:: python

    import bipartitepandas as bpd

* ``bipartitepandas.BipartiteLong``: Class for formatting bipartite networks in long form.

* ``bipartitepandas.BipartiteLongCollapsed``: Class for formatting bipartite networks in collapsed long form (i.e. employment spells are collapsed into a single observation).

* ``bipartitepandas.BipartiteEventStudy``: Class for formatting bipartite networks in event study form.

* ``bipartitepandas.BipartiteEventStudyCollapsed``: Class for formatting bipartite networks in collapsed event study form (i.e. employment spells are collapsed into a single observation).

Modules and Methods
-------------------

``bipartitepandas.BipartiteLong``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLong
   ~bipartitepandas.BipartiteData.clean_data
   ~bipartitepandas.BipartiteData.cluster
   ~bipartitepandas.BipartiteData.drop
   ~bipartitepandas.BipartiteData.get_collapsed_long
   ~bipartitepandas.BipartiteData.get_es
   ~bipartitepandas.BipartiteData.rename

``bipartitepandas.BipartiteLongCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autosummary::

   ~bipartitepandas.BipartiteLongCollapsed
   ~bipartitepandas.BipartiteLongCollapsed.clean_data
   ~bipartitepandas.BipartiteLongCollapsed.cluster
   ~bipartitepandas.BipartiteLongCollapsed.drop
   ~bipartitepandas.BipartiteLongCollapsed.get_es
   ~bipartitepandas.BipartiteLongCollapsed.rename

``bipartitepandas.BipartiteEventStudy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudy
   ~bipartitepandas.BipartiteEventStudy.clean_data
   ~bipartitepandas.BipartiteEventStudy.cluster
   ~bipartitepandas.BipartiteEventStudy.drop
   ~bipartitepandas.BipartiteEventStudy.get_long
   ~bipartitepandas.BipartiteEventStudy.rename

``bipartitepandas.BipartiteEventStudyCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudyCollapsed
   ~bipartitepandas.BipartiteEventStudyCollapsed.clean_data
   ~bipartitepandas.BipartiteEventStudyCollapsed.cluster
   ~bipartitepandas.BipartiteEventStudyCollapsed.drop
   ~bipartitepandas.BipartiteEventStudyCollapsed.get_collapsed_long
   ~bipartitepandas.BipartiteEventStudyCollapsed.rename
