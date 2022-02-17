Using from Python
=================

To install via pip, from the command line run::

   pip install bipartitepandas

To install via Conda, from the command line run::

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
   bdf = bpd.BipartiteDataFrame(i=df['i'], j=df['j'], y=df['j'], t=df['t'])
   # Clean your data
   bdf = bdf.clean()
   # Cluster
   bdf = bdf.cluster()

Check out the notebooks for more detailed examples!
