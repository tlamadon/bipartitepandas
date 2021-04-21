Using from Python
=================

To install from pip, from the command line run::

   pip install bipartitepandas

To install from conda, from the command line run::

   conda install -c tlamadon bipartitepandas

Sample data: :download:`download <twoway_sample_data.csv>`

To run in Python:

- If you want to create a BipartitePandas object:

.. code-block:: python

   import pandas as pd
   import bipartitepandas as bpd
   # Load data into pandas dataframe
   df = pd.read_csv(filepath)
   # Create bipartitepandas object
   bdf = bpd.BipartiteLong(df)
   # Clean your data
   bdf = bdf.clean_data()
   # Cluster
   bdf = bdf.cluster()

.. note::
   Your data must be long, collapsed long, event study, or collapsed event study.

   Long data must include the following columns:
    - ``i``: worker identifier
    - ``j``: firm identifier
    - ``y``: compensation
    - ``t``: time
   Long data may optionally include the following columns:
    - ``g``: firm cluster
    - ``m``: 1 if mover, 0 if stayer
   .. list-table:: Example long data
      :widths: 25 25 25 25
      :header-rows: 1
      :align: center

      * - i
        - j
        - y
        - t

      * - 1
        - 1
        - 1000
        - 2019
      * - 1
        - 2
        - 1500
        - 2020
      * - 2
        - 3
        - 500
        - 2019
      * - 2
        - 3
        - 550
        - 2020
   Collapsed long data must include the following columns:
    - ``i``: worker identifier
    - ``j``: firm identifier
    - ``y``: compensation averaged over the spell
    - ``t1``: first period in the spell
    - ``t2``: last period in the spell
   Collapsed long data may optionally include the following columns:
    - ``w``: weight of observation
    - ``g``: firm cluster
    - ``m``: 1 if mover, 0 if stayer
   .. list-table:: Example collapsed long data
      :widths: 20 20 20 20 20
      :header-rows: 1
      :align: center

      * - i
        - j
        - y
        - t1
        - t2

      * - 1
        - 1
        - 1000
        - 2019
        - 2019
      * - 1
        - 2
        - 1500
        - 2020
        - 2020
      * - 2
        - 3
        - 525
        - 2019
        - 2020
   Event study data must include the following columns:
    - ``i``: worker identifier
    - ``j1``: first firm identifier
    - ``j2``: second firm identifier
    - ``y1``: compensation for the first observation
    - ``y2``: compensation for the second observation
   Event study data may optionally include the following columns:
    - ``t1``: period of the first observation
    - ``t2``: period of the second observation
    - ``g1``: firm 1 cluster
    - ``g2``: firm 2 cluster
    - ``m``: 1 if mover, 0 if stayer
   .. list-table:: Example event study data
      :widths: 14 14 14 14 14 14 14
      :header-rows: 1
      :align: center

      * - i
        - j1
        - j2
        - y1
        - y2
        - t1
        - t2

      * - 1
        - 1
        - 2
        - 1000
        - 1500
        - 2019
        - 2020
      * - 2
        - 3
        - 3
        - 500
        - 500
        - 2019
        - 2019
      * - 2
        - 3
        - 3
        - 550
        - 550
        - 2020
        - 2020
   Collapsed event study data must include the following columns:
    - ``i``: worker identifier
    - ``j1``: first firm identifier
    - ``j2``: second firm identifier
    - ``y1``: compensation averaged over the first spell
    - ``y2``: compensation averaged over the second spell
   Collapsed event study data may optionally include the following columns:
    - ``t11``: first period in the first spell
    - ``t12``: last period in the first spell
    - ``t21``: first period in the second spell
    - ``t22``: last period in the second spell
    - ``w1``: weight of first spell
    - ``w2``: weight of second spell
    - ``g1``: firm 1 cluster
    - ``g2``: firm 2 cluster
    - ``m``: 1 if mover, 0 if stayer
   .. list-table:: Example collapsed event study data
      :widths: 11 11 11 11 11 11 11 11 11
      :header-rows: 1
      :align: center

      * - i
        - j1
        - j2
        - y1
        - y2
        - t11
        - t12
        - t21
        - t22

      * - 1
        - 1
        - 2
        - 1000
        - 1500
        - 2019
        - 2019
        - 2020
        - 2020
      * - 2
        - 3
        - 3
        - 525
        - 525
        - 2019
        - 2020
        - 2019
        - 2020
