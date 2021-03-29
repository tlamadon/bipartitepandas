Using from python
=================

To install from pip, from the command line run::

   pip install bipartitepandas

Sample data: :download:`download <twoway_sample_data.csv>`

To run in python:

- If you want to create a bipartitepandas object:

.. code-block:: python

   import bipartitepandas as bpd
   # Create bipartitepandas object
   df = bpd.BipartiteLong(data)
   # Clean your data
   df.clean_data()
   # Cluster
   df.cluster()

.. note::
   Your data must be long, collapsed long, event study, or collapsed event study.
   Long data must include the following columns:
    - ``wid``: the worker identifier
    - ``fid``: the firm identifier
    - ``year``: the time
    - ``comp``: the outcome variable, in our case compensation
   Long data may optionally include the following columns:
    - ``m``: 1 if mover, 0 if stayer
    - ``j``: firm cluster
   .. list-table:: Example long data
      :widths: 25 25 25 25
      :header-rows: 1
      :align: center

      * - wid
        - fid
        - year
        - comp

      * - 1
        - 1
        - 2019
        - 1000
      * - 1
        - 2
        - 2020
        - 1500
      * - 2
        - 3
        - 2019
        - 500
      * - 2
        - 3
        - 2020
        - 550
   Collapsed long data must include the following columns:
    - ``wid``: the worker identifier
    - ``fid``: the firm identifier
    - ``year_start``: first time in the spell
    - ``year_end``: last time in the spell
    - ``comp``: the outcome variable averaged over the spell, in our case compensation
   Collapsed long data may optionally include the following columns:
    - ``m``: 1 if mover, 0 if stayer
    - ``j``: firm cluster
    - ``weight``: weight of observation
   .. list-table:: Example collapsed long data
      :widths: 20 20 20 20 20
      :header-rows: 1
      :align: center

      * - wid
        - fid
        - year_start
        - year_end
        - comp

      * - 1
        - 1
        - 2019
        - 2019
        - 1000
      * - 1
        - 2
        - 2020
        - 2020
        - 1500
      * - 2
        - 3
        - 2020
        - 525
   Event study data must include the following columns:
    - ``wid``: the worker identifier
    - ``f1i``: the first firm identifier
    - ``f2i``: the second firm identifier
    - ``y1``: the outcome variable for the first observation, in our case compensation
    - ``y2``: the outcome variable for the second observation, in our case compensation
   Event study data may optionally include the following columns:
    - ``year_1``: the time for the first observation
    - ``year_2``: the time for the second observation
    - ``m``: 1 if mover, 0 if stayer
    - ``j1``: firm 1 cluster
    - ``j2``: firm 2 cluster
   .. list-table:: Example event study data
      :widths: 14 14 14 14 14 14 14
      :header-rows: 1
      :align: center

      * - wid
        - f1i
        - f2i
        - year_1
        - year_2
        - y1
        - y2

      * - 1
        - 1
        - 2
        - 2019
        - 2020
        - 1000
        - 1500
      * - 2
        - 3
        - 3
        - 2019
        - 2020
        - 500
        - 550
   Collapsed event study data must include the following columns:
    - ``wid``: the worker identifier
    - ``f1i``: the first firm identifier
    - ``f2i``: the second firm identifier
    - ``y1``: the outcome variable averaged over the spell for the first observation, in our case compensation
    - ``y2``: the outcome variable averaged over the spell for the second observation, in our case compensation
   Collapsed event study data may optionally include the following columns:
    - ``year_start_1``: first time in the first spell
    - ``year_end_1``: last time in the first spell
    - ``year_start_2``: first time in the second spell
    - ``year_end_2``: last time in the second spell
    - ``m``: 1 if mover, 0 if stayer
    - ``j1``: firm 1 cluster
    - ``j2``: firm 2 cluster
    - ``w1``: weight of first spell
    - ``w2``: weight of second spell
   .. list-table:: Example collapsed event study data
      :widths: 11 11 11 11 11 11 11 11 11
      :header-rows: 1
      :align: center

      * - wid
        - f1i
        - f2i
        - year_start_1
        - year_end_1
        - year_start_2
        - year_end_2
        - y1
        - y2

      * - 1
        - 1
        - 2
        - 2019
        - 2019
        - 2020
        - 2020
        - 1000
        - 1500
      * - 2
        - 3
        - 3
        - 2019
        - 2020
        - 2019
        - 2020
        - 525
        - 525
