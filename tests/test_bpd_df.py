'''
Tests for bipartitepandas.
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pickle

########################################
##### Tests for BipartiteDataFrame #####
########################################

def test_dataframe_1():
    # Test BipartiteDataFrame constructor for different formats

    ## Long format ##
    a = bpd.SimBipartite().simulate()
    print('a.columns:', a.columns)
    b = bpd.BipartiteDataFrame(**a).clean()

    assert isinstance(b, bpd.BipartiteLong)
    for col in ['l', 'k', 'alpha', 'psi']:
        assert col in b.col_reference_dict.keys()
    for col in ['l', 'k']:
        assert b.col_dtype_dict[col] == 'float' # 'int'
    for col in ['alpha', 'psi']:
        assert b.col_dtype_dict[col] == 'float'

    ## Long collapsed format ##
    a2 = pd.DataFrame(b.collapse())
    b2 = bpd.BipartiteDataFrame(**a2).clean()

    assert isinstance(b2, bpd.BipartiteLongCollapsed)
    assert b2.col_reference_dict['t'] == ['t1', 't2']

    ## Event study format ##
    a3 = pd.DataFrame(b.to_eventstudy())
    b3 = bpd.BipartiteDataFrame(**a3).clean()

    assert isinstance(b3, bpd.BipartiteEventStudy)
    for col in ['j', 'y', 't', 'l', 'k', 'alpha', 'psi']:
        assert b3.col_reference_dict[col] == [col + '1', col + '2']

    ## Event study collapsed format ##
    a4 = pd.DataFrame(b2.to_eventstudy())
    b4 = bpd.BipartiteDataFrame(**a4).clean()

    assert isinstance(b4, bpd.BipartiteEventStudyCollapsed)
    assert b4.col_reference_dict['t'] == ['t11', 't12', 't21', 't22']
    for col in ['j', 'y', 'l', 'k', 'alpha', 'psi']:
        assert b4.col_reference_dict[col] == [col + '1', col + '2']

def test_dataframe_2():
    # Test BipartiteDataFrame constructor with custom attributes dictionaries

    ## Long format ##
    a = bpd.SimBipartite().simulate()
    b = bpd.BipartiteDataFrame(**a, custom_categorical_dict={'l': True}, custom_dtype_dict={'l': 'categorical'}, custom_how_collapse_dict={'alpha': None, 'l': None}, custom_long_es_split_dict={'psi': False}).clean()

    assert 'l' in b.columns_contig.keys()
    assert b.col_dtype_dict['l'] == 'categorical'
    assert b.col_collapse_dict['alpha'] is None
    assert b.col_long_es_dict['psi'] is False
