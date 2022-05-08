'''
Tests for bipartitepandas.
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pickle

#####################################
##### Tests for BipartitePandas #####
#####################################

def test_reformatting_1():
    # Convert from long --> event study --> long --> collapsed long --> collapsed event study --> collapsed long to ensure conversion maintains data properly.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'g': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2, 'g': 2})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'g': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'g': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 2., 't': 3, 'g': 1})
    worker_data.append({'i': 1, 'j': 5, 'y': 1., 't': 3, 'g': 3})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1, 'g': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2, 'g': 1})
    # Worker 4
    worker_data.append({'i': 4, 'j': 0, 'y': 1., 't': 1, 'g': 1})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.to_eventstudy()
    bdf = bdf.clean()
    bdf = bdf.to_long()
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.clean()
    bdf = bdf.to_eventstudy()
    bdf = bdf.clean()
    bdf = bdf.to_long()
    bdf = bdf.clean()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j'] == 2
    assert stayers.iloc[0]['y'] == 1
    assert stayers.iloc[0]['t1'] == 1
    assert stayers.iloc[0]['t2'] == 2
    assert stayers.iloc[0]['g'] == 0

    assert stayers.iloc[1]['i'] == 3
    assert stayers.iloc[1]['j'] == 0
    assert stayers.iloc[1]['y'] == 1
    assert stayers.iloc[1]['t1'] == 1
    assert stayers.iloc[1]['t2'] == 1
    assert stayers.iloc[1]['g'] == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['y'] == 2
    assert movers.iloc[0]['t1'] == 1
    assert movers.iloc[0]['t2'] == 1
    assert movers.iloc[0]['g'] == 0

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['y'] == 1
    assert movers.iloc[1]['t1'] == 2
    assert movers.iloc[1]['t2'] == 2
    assert movers.iloc[1]['g'] == 1

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['y'] == 1
    assert movers.iloc[2]['t1'] == 1
    assert movers.iloc[2]['t2'] == 1
    assert movers.iloc[2]['g'] == 1

    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['j'] == 2
    assert movers.iloc[3]['y'] == 1.5
    assert movers.iloc[3]['t1'] == 2
    assert movers.iloc[3]['t2'] == 3
    assert movers.iloc[3]['g'] == 0
