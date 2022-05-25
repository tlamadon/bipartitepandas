'''
Tests for bipartitepandas.
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pickle

####################################
##### Tests for Custom Columns #####
####################################

def test_custom_columns_1():
    # Make sure custom columns are being constructed properly, and work for long-event study converseions
    worker_data = []
    # Firm 0 -> 1 -> 0
    # Time 1 -> 2 -> 4
    # Custom 0 -> 2 -> 3
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'c': 0})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2, 'c': 2})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4, 'c': 3})
    # Firm 1 -> 2
    # Custom 3 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'c': 3})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'c': 2})
    # Firm 2 -> 1
    # Custom 2 -> 0
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1, 'c': 2})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2, 'c': 0})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    ## First, try constructing without adding column ##
    try:
        bdf = bpd.BipartiteLong(data=df).clean()
        success = False
    except ValueError:
        success = True

    assert success

    ## 1.5, try constructing with column that has digit in name ##
    bdf = bpd.BipartiteLong(data=df.rename({'c': 'c1'}, axis=1))
    try:
        bdf = bdf.add_column('c1')
        success = False
    except NotImplementedError:
        success = True

    assert success

    ## Second, construct while adding column, but don't make it categorical ##
    bdf = bpd.BipartiteLong(data=df).add_column('c').clean()

    assert 'c' in bdf.columns
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()
    assert bdf.n_unique_ids('c') != bdf['c'].max() + 1

    ## Third, construct while adding column, and make it categorical ##
    bdf = bpd.BipartiteLong(data=df, track_id_changes=True).add_column('c', is_categorical=True, dtype='categorical').clean()

    assert 'c' in bdf.columns
    assert 'c' in bdf.id_reference_dict.keys()
    assert 'c' in bdf.original_ids().columns
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()
    assert bdf.n_unique_ids('c') == bdf['c'].max() + 1

    ## Fourth, construct while adding column, and make it categorical but with an invalid collapse option ##
    try:
        bdf = bpd.BipartiteLong(data=df, track_id_changes=True).add_column('c', is_categorical=True, dtype='categorical', how_collapse='mean').clean()
        success = False
    except NotImplementedError:
        success = True

    assert success

    ## Fifth, try adding column with no associated data ##
    try:
        bdf = bpd.BipartiteLong(data=df).add_column('r').clean()
        success = False
    except ValueError:
        success = True

    assert success

    ## Sixth, try adding column where listed subcolumns are already assigned ##
    try:
        bdf = bpd.BipartiteLong(data=df).add_column('c', col_reference='i').clean()
        success = False
    except ValueError:
        success = True

    assert success

    ## Seventh, try with event study format, but don't make custom columns categorical ##
    bdf = bpd.BipartiteEventStudy(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c').clean().to_eventstudy())).add_column('c').clean()

    assert 'c1' in bdf.columns and 'c2' in bdf.columns
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()
    assert bdf.n_unique_ids('c') != max(bdf['c1'].max(), bdf['c2'].max()) + 1

    ## Eighth, try with event study format, and make custom columns categorical ##
    bdf = bpd.BipartiteEventStudy(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c').clean().to_eventstudy()), track_id_changes=True).add_column('c', is_categorical=True, dtype='categorical').clean()

    assert 'c1' in bdf.columns and 'c2' in bdf.columns
    assert 'c' in bdf.id_reference_dict.keys()
    assert ('c1' in bdf.original_ids().columns) and ('c2' in bdf.original_ids().columns)
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()
    assert bdf.n_unique_ids('c') == max(bdf['c1'].max(), bdf['c2'].max()) + 1

    ## Ninth, try with event study format, but don't mark custom column as not being split ##
    try:
        bdf = bpd.BipartiteEventStudy(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c', long_es_split=False).clean().to_eventstudy())).add_column('c').clean()
        success = False
    except ValueError:
        success = True

    assert success

    ## Tenth, try with event study format, but don't split custom column (set to None for event study to make sure data cleaning handles this case properly) ##
    bdf = bpd.BipartiteEventStudy(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c', long_es_split=False).clean().to_eventstudy())).add_column('c', long_es_split=None).clean()

    assert 'c' in bdf.columns
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()
    assert bdf.n_unique_ids('c') != bdf['c'].max() + 1

    ## Eleventh, go from event study and back to long ##
    bdf = bpd.BipartiteEventStudy(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c').clean().to_eventstudy())).add_column('c').clean().to_long()

    assert 'c' in bdf.columns
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()
    assert bdf.n_unique_ids('c') != bdf['c'].max() + 1

    ## Twelfth, go from long to event study with categorical column that should drop ##
    bdf = bpd.BipartiteLong(data=df, track_id_changes=True).add_column('c', is_categorical=True, dtype='categorical', long_es_split=None).clean().to_eventstudy()

    assert ('c1' not in bdf.columns) and ('c2' not in bdf.columns)
    assert 'c' not in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.id_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' not in bdf.col_dtype_dict.keys()
    assert 'c' not in bdf.col_collapse_dict.keys()
    assert 'c' not in bdf.col_long_es_dict.keys()

    ## Thirteenth, go from event study to long with categorical column that should drop ##
    bdf = bpd.BipartiteEventStudy(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c').clean().to_eventstudy()), track_id_changes=True).add_column('c', is_categorical=True, dtype='categorical', long_es_split=None).clean().to_long()

    assert ('c1' not in bdf.columns) and ('c2' not in bdf.columns)
    assert 'c' not in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.id_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' not in bdf.col_dtype_dict.keys()
    assert 'c' not in bdf.col_collapse_dict.keys()
    assert 'c' not in bdf.col_long_es_dict.keys()

def test_custom_columns_2():
    # Make sure custom columns work for long-collapsed long conversions
    worker_data = []
    # Firm 0 -> 0 -> 1 -> 0
    # Custom 0 -> 2 -> 2 -> 3
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'c': 0})
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 2, 'c': 1.5})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 3, 'c': 2})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4, 'c': 3})
    # Firm 1 -> 2
    # Custom 3 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'c': 3})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'c': 2})
    # Firm 2 -> 1
    # Custom 2 -> 0
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1, 'c': 2})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2, 'c': 0})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    ## First, collapse by first ##
    bdf = bpd.BipartiteLong(data=df).add_column('c', how_collapse='first').clean().collapse()

    assert bdf.iloc[0]['c'] == 0

    ## Second, collapse by last ##
    bdf = bpd.BipartiteLong(data=df).add_column('c', how_collapse='last').clean().collapse()

    assert bdf.iloc[0]['c'] == 1.5

    ## Third, collapse by mean ##
    bdf = bpd.BipartiteLong(data=df).add_column('c', how_collapse='mean').clean().collapse()

    assert bdf.iloc[0]['c'] == 0.75

    ## Fourth, collapse by None ##
    bdf = bpd.BipartiteLong(data=df, track_id_changes=True).add_column('c', is_categorical=True, dtype='categorical', how_collapse=None).clean().collapse()

    assert 'c' not in bdf.columns
    assert 'c' not in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.id_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' not in bdf.col_dtype_dict.keys()
    assert 'c' not in bdf.col_collapse_dict.keys()
    assert 'c' not in bdf.col_long_es_dict.keys()

    ## Fifth, collapse then uncollapse by first ##
    bdf = bpd.BipartiteLong(data=df).add_column('c', how_collapse='first').clean().collapse().uncollapse()

    assert bdf.iloc[0]['c'] == 0
    assert bdf.iloc[1]['c'] == 0

    ## Sixth, collapse then uncollapse by last ##
    bdf = bpd.BipartiteLong(data=df).add_column('c', how_collapse='last').clean().collapse().uncollapse()

    assert bdf.iloc[0]['c'] == 1.5
    assert bdf.iloc[1]['c'] == 1.5

    ## Seventh, collapse then uncollapse by mean ##
    bdf = bpd.BipartiteLong(data=df).add_column('c', how_collapse='mean').clean().collapse().uncollapse()

    assert bdf.iloc[0]['c'] == 0.75
    assert bdf.iloc[1]['c'] == 0.75

    ## Eighth, clean collapsed with None ##
    bdf = bpd.BipartiteLongCollapsed(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c', how_collapse='mean').clean().collapse()), track_id_changes=True).add_column('c', is_categorical=True, dtype='categorical', how_collapse=None).clean()

    assert 'c' in bdf.columns
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' in bdf.id_reference_dict.keys()
    assert 'c' in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()

    ## Ninth, uncollapse by None ##
    bdf = bpd.BipartiteLongCollapsed(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c', how_collapse='mean').clean().collapse()), track_id_changes=True).add_column('c', is_categorical=True, dtype='categorical', how_collapse=None).clean().uncollapse()

    assert 'c' not in bdf.columns
    assert 'c' not in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.id_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' not in bdf.col_dtype_dict.keys()
    assert 'c' not in bdf.col_collapse_dict.keys()
    assert 'c' not in bdf.col_long_es_dict.keys()

def test_custom_columns_3():
    # Make sure setting column properties works properly
    worker_data = []
    # Firm 0 -> 0 -> 1 -> 0
    # Custom 0 -> 2 -> 2 -> 3
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'c': 0})
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 2, 'c': 1.5})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 3, 'c': 2})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4, 'c': 3})
    # Firm 1 -> 2
    # Custom 3 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'c': 3})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'c': 2})
    # Firm 2 -> 1
    # Custom 2 -> 0
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1, 'c': 2})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2, 'c': 0})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    ## First, categorical to not categorical ##
    bdf = bpd.BipartiteLong(data=df, track_id_changes=True).add_column('c', is_categorical=True, dtype='categorical').clean()
    init_properties = bdf.get_column_properties('c')

    assert (init_properties['dtype'] == 'categorical') and init_properties['is_categorical'] and (init_properties['how_collapse'] == 'first') and init_properties['long_es_split']
    assert ('c' in bdf.columns_contig.keys()) and ('c' in bdf.id_reference_dict.keys())

    bdf = bdf.set_column_properties('c', dtype='int', is_categorical=False, how_collapse='mean', long_es_split=None)
    new_properties = bdf.get_column_properties('c')

    assert (new_properties['dtype'] == 'int') and (not new_properties['is_categorical']) and (new_properties['how_collapse'] == 'mean') and (new_properties['long_es_split'] is None)
    assert ('c' not in bdf.columns_contig.keys()) and ('c' not in bdf.id_reference_dict.keys())

    ## Second, not categorical to categorical ##
    bdf = bpd.BipartiteLong(data=df, track_id_changes=True).add_column('c', is_categorical=False).clean()
    init_properties = bdf.get_column_properties('c')

    assert (init_properties['dtype'] == 'any') and (not init_properties['is_categorical']) and (init_properties['how_collapse'] == 'first') and init_properties['long_es_split']
    assert ('c' not in bdf.columns_contig.keys()) and ('c' not in bdf.id_reference_dict.keys())

    bdf = bdf.set_column_properties('c', is_categorical=True, dtype='categorical', how_collapse='last', long_es_split=None)
    new_properties = bdf.get_column_properties('c')

    assert (new_properties['dtype'] == 'categorical') and new_properties['is_categorical'] and (new_properties['how_collapse'] == 'last') and (new_properties['long_es_split'] is None)
    assert ('c' in bdf.columns_contig.keys()) and ('c' in bdf.id_reference_dict.keys())

    ## Third, not categorical to categorical, but invalid how_collapse ##
    bdf = bpd.BipartiteLong(data=df, track_id_changes=True).add_column('c', is_categorical=False).clean()
    init_properties = bdf.get_column_properties('c')

    try:
        bdf = bdf.set_column_properties('c', is_categorical=True, dtype='categorical', how_collapse='mean', long_es_split=None)
        success = False
    except NotImplementedError:
        success = True

    assert success

def test_custom_columns_4():
    # Make sure rename works properly with custom columns
    worker_data = []
    # Firm 0 -> 0 -> 1 -> 0
    # Custom 0 -> 2 -> 2 -> 3
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'c': 0})
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 2, 'c': 1.5})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 3, 'c': 2})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4, 'c': 3})
    # Firm 1 -> 2
    # Custom 3 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'c': 3})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'c': 2})
    # Firm 2 -> 1
    # Custom 2 -> 0
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1, 'c': 2})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2, 'c': 0})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    ## Categorical column should alter id_reference_dict properly ##
    bdf = bpd.BipartiteLong(data=df, track_id_changes=True).add_column('c', is_categorical=True, dtype='categorical').clean()

    assert 'c' in bdf.id_reference_dict.keys()

    bdf = bdf.rename({'c': 'cw'}, axis=1)

    assert 'cw' in bdf.id_reference_dict.keys()

    try:
        bdf = bdf.rename({'cw': 'cw1'}, axis=1)
        success = False
    except NotImplementedError:
        success = True

    assert success
