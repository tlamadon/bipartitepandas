'''
Tests for bipartitepandas.
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pickle

############################################
##### Tests for BipartiteLongCollapsed #####
############################################

def test_long_collapsed_1():
    # Test constructor for BipartiteLongCollapsed.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    df = pd.DataFrame(bdf.collapse())
    bdf = bpd.BipartiteLongCollapsed(df)
    bdf = bdf.clean()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j'] == 2
    assert stayers.iloc[0]['y'] == 1

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['y'] == 2

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['y'] == 1

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['y'] == 1

    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['j'] == 2
    assert movers.iloc[3]['y'] == 1

def test_collapsed_weights_2():
    # Test that weights are computed correctly.
    a = bpd.SimBipartite().simulate(np.random.default_rng(1234))
    # Non-collapsed data
    b = bpd.BipartiteDataFrame(a.copy()).clean()
    # Collapsed data
    c = bpd.BipartiteDataFrame(a.copy()).clean().collapse().to_eventstudy().to_long()

    assert len(c) < len(b)
    assert np.sum(c.loc[:, 'w'].to_numpy()) == len(b)
