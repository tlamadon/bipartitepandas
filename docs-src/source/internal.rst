==========
Python API
==========

Overview
---------

The main bipartitepandas API is split into four classes. bipartitepandas is canonically imported using

  .. code-block:: python

    import bipartitepandas as bpd

* ``pytwoway.BipartiteLong``: Class for formatting bipartite networks in long form.

* ``pytwoway.BipartiteLongCollapsed``: Class for formatting bipartite networks in collapsed long form (i.e. employment spells are collapsed into a single observation).

* ``pytwoway.BipartiteEventStudy``: Class for formatting bipartite networks in event study form.

* ``pytwoway.BipartiteEventStudyCollapsed``: Class for formatting bipartite networks in collapsed event study form (i.e. employment spells are collapsed into a single observation).

Modules and Methods
-------------------

``pytwoway.BipartiteLong``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteLong
   ~bipartitepandas.BipartiteData.clean_data
   ~bipartitepandas.BipartiteData.cluster
   ~bipartitepandas.BipartiteData.drop
   ~bipartitepandas.BipartiteData.get_collapsed_long
   ~bipartitepandas.BipartiteData.get_es
   ~bipartitepandas.BipartiteData.rename

``pytwoway.BipartiteLongCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autosummary::

   ~bipartitepandas.BipartiteLongCollapsed
   ~bipartitepandas.BipartiteLongCollapsed.clean_data
   ~bipartitepandas.BipartiteLongCollapsed.cluster
   ~bipartitepandas.BipartiteLongCollapsed.drop
   ~bipartitepandas.BipartiteLongCollapsed.get_es
   ~bipartitepandas.BipartiteLongCollapsed.rename

``pytwoway.BipartiteEventStudy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudy
   ~bipartitepandas.BipartiteEventStudy.clean_data
   ~bipartitepandas.BipartiteEventStudy.cluster
   ~bipartitepandas.BipartiteEventStudy.drop
   ~bipartitepandas.BipartiteEventStudy.get_long
   ~bipartitepandas.BipartiteEventStudy.rename

``pytwoway.BipartiteEventStudyCollapsed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~bipartitepandas.BipartiteEventStudyCollapsed
   ~bipartitepandas.BipartiteEventStudyCollapsed.clean_data
   ~bipartitepandas.BipartiteEventStudyCollapsed.cluster
   ~bipartitepandas.BipartiteEventStudyCollapsed.drop
   ~bipartitepandas.BipartiteEventStudyCollapsed.get_collapsed_long
   ~bipartitepandas.BipartiteEventStudyCollapsed.rename
