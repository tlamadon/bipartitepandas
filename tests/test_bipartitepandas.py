'''
Tests for bipartitepandas

DATE: March 2021
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd

###################################
##### Tests for BipartiteBase #####
###################################

def test_refactor_1():
    # Continuous time, 2 movers between firms 1 and 2, and 1 stayer at firm 3, and discontinuous time still counts as a move.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 0, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 2, 'y': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.shape[0] == 0

def test_refactor_2():
    # Discontinuous time, 2 movers between firms 1 and 2, and 1 stayer at firm 3, and discontinuous time still counts as a move.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 3, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 0, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 2, 'y': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

def test_refactor_3():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'index': 4})
    worker_data.append({'j': 1, 't': 2, 'i': 2, 'y': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['j1'] == 2
    assert movers.iloc[2]['j2'] == 1
    assert movers.iloc[2]['i'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 2

def test_refactor_4():
    # Continuous time, 1 mover between firms 1 and 2 and then 2 and 1, 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2, 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1, 'index': 1})
    worker_data.append({'j': 0, 't': 3, 'i': 0, 'y': 1, 'index': 2})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1, 'index': 3})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1, 'index': 4})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1, 'index': 5})
    worker_data.append({'j': 1, 't': 2, 'i': 2, 'y': 2, 'index': 6})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['j1'] == 1
    assert movers.iloc[2]['j2'] == 2
    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_5():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1, 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 0, 't': 4, 'i': 0, 'y': 1., 'index': 2})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'index': 5})
    worker_data.append({'j': 1, 't': 2, 'i': 2, 'y': 2., 'index': 6})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['j1'] == 1
    assert movers.iloc[2]['j2'] == 2
    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_6():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are continuous), 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 0, 't': 2, 'i': 0, 'y': 2., 'index': 1})
    worker_data.append({'j': 1, 't': 3, 'i': 0, 'y': 1., 'index': 2})
    worker_data.append({'j': 0, 't': 5, 'i': 0, 'y': 1., 'index': 3})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 5})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'index': 6})
    worker_data.append({'j': 1, 't': 2, 'i': 2, 'y': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['j1'] == 1
    assert movers.iloc[2]['j2'] == 2
    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_7():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are discontinuous), 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 0, 't': 3, 'i': 0, 'y': 2., 'index': 1})
    worker_data.append({'j': 1, 't': 4, 'i': 0, 'y': 1., 'index': 2})
    worker_data.append({'j': 0, 't': 6, 'i': 0, 'y': 1., 'index': 3})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 5})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'index': 6})
    worker_data.append({'j': 1, 't': 2, 'i': 2, 'y': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['j1'] == 1
    assert movers.iloc[2]['j2'] == 2
    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_8():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are discontinuous), 1 between firms 1 and 2, and 1 between firms 3 and 2, and discontinuous time still counts as a move.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 0, 't': 3, 'i': 0, 'y': 2., 'index': 1})
    worker_data.append({'j': 1, 't': 4, 'i': 0, 'y': 1., 'index': 2})
    worker_data.append({'j': 0, 't': 6, 'i': 0, 'y': 1., 'index': 3})
    worker_data.append({'j': 0, 't': 1, 'i': 1, 'y': 1., 'index': 4})
    worker_data.append({'j': 1, 't': 2, 'i': 1, 'y': 1., 'index': 5})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'index': 6})
    worker_data.append({'j': 1, 't': 2, 'i': 2, 'y': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['j1'] == 0
    assert movers.iloc[2]['j2'] == 1
    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_9():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are discontinuous), 1 between firms 2 and 1, and 1 between firms 3 and 2, and discontinuous time still counts as a move.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 0, 't': 3, 'i': 0, 'y': 2., 'index': 1})
    worker_data.append({'j': 1, 't': 4, 'i': 0, 'y': 1., 'index': 2})
    worker_data.append({'j': 0, 't': 6, 'i': 0, 'y': 1., 'index': 3})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 4})
    worker_data.append({'j': 0, 't': 2, 'i': 1, 'y': 1., 'index': 5})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'index': 6})
    worker_data.append({'j': 1, 't': 2, 'i': 2, 'y': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['j1'] == 1
    assert movers.iloc[2]['j2'] == 0
    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_10():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 3, and 1 stayer at firm 3, and discontinuous time still counts as a move.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 2, 'y': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

def test_contiguous_fids_11():
    # Check contiguous_ids() with firm ids.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 3, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 3, 't': 1, 'i': 2, 'y': 1., 'index': 4})
    worker_data.append({'j': 3, 't': 2, 'i': 2, 'y': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1


def test_contiguous_wids_12():
    # Check contiguous_ids() with worker ids.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

def test_contiguous_cids_13():
    # Check contiguous_ids() with cluster ids.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'g': 1, 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'g': 2, 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'g': 2, 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'g': 1, 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'g': 1, 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 2, 'y': 1., 'g': 1, 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't', 'g']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1
    assert movers.iloc[0]['g1'] == 0
    assert movers.iloc[0]['g2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1
    assert movers.iloc[1]['g1'] == 1
    assert movers.iloc[1]['g2'] == 0

    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['g1'] == 0
    assert stayers.iloc[0]['g2'] == 0

def test_contiguous_cids_14():
    # Check contiguous_ids() with cluster ids.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'g': 2, 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'g': 1, 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'g': 1, 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'g': 2, 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'g': 2, 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 2, 'y': 1., 'g': 2, 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't', 'g']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1
    assert movers.iloc[0]['g1'] == 1
    assert movers.iloc[0]['g2'] == 0

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1
    assert movers.iloc[1]['g1'] == 0
    assert movers.iloc[1]['g2'] == 1

    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['g1'] == 1
    assert stayers.iloc[0]['g2'] == 1

def test_col_dict_15():
    # Check that col_dict works properly.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']].rename({'j': 'firm', 'i': 'worker'}, axis=1)

    bdf = bpd.BipartiteLong(data=df, col_dict={'j': 'firm', 'i': 'worker'})
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

def test_worker_year_unique_16_1():
    # Workers with multiple jobs in the same year, keep the highest paying, with long format. Testing 'max', 'sum', and 'mean' options, where options should not have an effect.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 3, 't': 2, 'i': 1, 'y': 0.5, 'index': 4})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 5})
    worker_data.append({'j': 1, 't': 1, 'i': 3, 'y': 1.5, 'index': 6})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    for how in ['max', 'sum', 'mean']:
        bdf = bpd.BipartiteLong(data=df)
        bdf = bdf.clean_data({'i_t_how': how}).gen_m()

        stayers = bdf[bdf['m'] == 0]
        movers = bdf[bdf['m'] == 1]

        assert movers.iloc[0]['i'] == 0
        assert movers.iloc[0]['j'] == 0
        assert movers.iloc[0]['y'] == 2
        assert movers.iloc[0]['t'] == 1

        assert movers.iloc[1]['i'] == 0
        assert movers.iloc[1]['j'] == 1
        assert movers.iloc[1]['y'] == 1
        assert movers.iloc[1]['t'] == 2

        assert movers.iloc[2]['i'] == 1
        assert movers.iloc[2]['j'] == 1
        assert movers.iloc[2]['y'] == 1
        assert movers.iloc[2]['t'] == 1

        assert movers.iloc[3]['i'] == 1
        assert movers.iloc[3]['j'] == 2
        assert movers.iloc[3]['y'] == 1
        assert movers.iloc[3]['t'] == 2

        assert movers.iloc[4]['i'] == 2
        assert movers.iloc[4]['j'] == 1
        assert movers.iloc[4]['y'] == 1.5
        assert movers.iloc[4]['t'] == 1

        assert movers.iloc[5]['i'] == 2
        assert movers.iloc[5]['j'] == 2
        assert movers.iloc[5]['y'] == 1
        assert movers.iloc[5]['t'] == 2

def test_worker_year_unique_16_2():
    # Workers with multiple jobs in the same year, keep the highest paying, with long format. Testing 'max', 'sum' and 'mean' options, where options should have an effect.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2.})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1.})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1.})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1.})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1.5})
    worker_data.append({'j': 3, 't': 2, 'i': 1, 'y': 0.5})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1.})
    worker_data.append({'j': 1, 't': 1, 'i': 3, 'y': 1.5})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1.})

    df = pd.concat([pd.DataFrame(worker, index=np.arange(len(worker_data))) for worker in worker_data])[['i', 'j', 'y', 't']]

    for how in ['max', 'sum', 'mean']:
        bdf = bpd.BipartiteLong(data=df)
        bdf = bdf.clean_data({'i_t_how': how}).gen_m()

        stayers = bdf[bdf['m'] == 0]
        movers = bdf[bdf['m'] == 1]

        assert movers.iloc[0]['i'] == 0
        assert movers.iloc[0]['j'] == 0
        assert movers.iloc[0]['y'] == 2
        assert movers.iloc[0]['t'] == 1

        assert movers.iloc[1]['i'] == 0
        assert movers.iloc[1]['j'] == 1
        assert movers.iloc[1]['y'] == 1
        assert movers.iloc[1]['t'] == 2

        assert movers.iloc[2]['i'] == 1
        assert movers.iloc[2]['j'] == 1
        assert movers.iloc[2]['y'] == 1
        assert movers.iloc[2]['t'] == 1

        assert movers.iloc[3]['i'] == 1
        assert movers.iloc[3]['j'] == 2
        if how == 'max':
            assert movers.iloc[3]['y'] == 1.5
        elif how == 'sum':
            assert movers.iloc[3]['y'] == 2.5
        elif how == 'mean':
            assert movers.iloc[3]['y'] == 1.25
        assert movers.iloc[3]['t'] == 2

        assert movers.iloc[4]['i'] == 2
        assert movers.iloc[4]['j'] == 1
        assert movers.iloc[4]['y'] == 1.5
        assert movers.iloc[4]['t'] == 1

        assert movers.iloc[5]['i'] == 2
        assert movers.iloc[5]['j'] == 2
        assert movers.iloc[5]['y'] == 1
        assert movers.iloc[5]['t'] == 2

def test_worker_year_unique_16_3():
    # Workers with multiple jobs in the same year, keep the highest paying, with collapsed long format. Testing 'max', 'sum', and 'mean' options, where options should have an effect.
    worker_data = []
    worker_data.append({'j': 0, 't1': 1, 't2': 1, 'i': 0, 'y': 2.})
    worker_data.append({'j': 1, 't1': 2, 't2': 2, 'i': 0, 'y': 1.})
    worker_data.append({'j': 1, 't1': 1, 't2': 2, 'i': 1, 'y': 1.})
    worker_data.append({'j': 2, 't1': 2, 't2': 2, 'i': 1, 'y': 1.})
    worker_data.append({'j': 2, 't1': 2, 't2': 2, 'i': 1, 'y': 1.5})
    worker_data.append({'j': 3, 't1': 2, 't2': 2, 'i': 1, 'y': 0.5})
    worker_data.append({'j': 2, 't1': 1, 't2': 2, 'i': 3, 'y': 1.})
    worker_data.append({'j': 1, 't1': 1, 't2': 2, 'i': 3, 'y': 1.5})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])[['i', 'j', 'y', 't1', 't2']]

    for how in ['max', 'sum', 'mean']:
        bdf = bpd.BipartiteLongCollapsed(data=df)
        bdf = bdf.clean_data({'i_t_how': how}).gen_m()

        stayers = bdf[bdf['m'] == 0]
        movers = bdf[bdf['m'] == 1]

        assert movers.iloc[0]['i'] == 0
        assert movers.iloc[0]['j'] == 0
        assert movers.iloc[0]['y'] == 2
        assert movers.iloc[0]['t1'] == 1
        assert movers.iloc[0]['t2'] == 1

        assert movers.iloc[1]['i'] == 0
        assert movers.iloc[1]['j'] == 1
        assert movers.iloc[1]['y'] == 1
        assert movers.iloc[1]['t1'] == 2
        assert movers.iloc[1]['t2'] == 2

        assert movers.iloc[2]['i'] == 1
        assert movers.iloc[2]['j'] == 1
        assert movers.iloc[2]['y'] == 1
        assert movers.iloc[2]['t1'] == 1
        assert movers.iloc[2]['t2'] == 1

        assert movers.iloc[3]['i'] == 1
        assert movers.iloc[3]['j'] == 2
        if how == 'max':
            assert movers.iloc[3]['y'] == 1.5
        elif how == 'sum':
            assert movers.iloc[3]['y'] == 2.5
        elif how == 'mean':
            assert movers.iloc[3]['y'] == 1.25
        assert movers.iloc[3]['t1'] == 2
        assert movers.iloc[3]['t2'] == 2

        assert stayers.iloc[0]['i'] == 2
        assert stayers.iloc[0]['j'] == 1
        assert stayers.iloc[0]['y'] == 1.5
        assert stayers.iloc[0]['t1'] == 1
        assert stayers.iloc[0]['t2'] == 2

def test_worker_year_unique_16_4():
    # Workers with multiple jobs in the same year, keep the highest paying, with event study format. Testing 'max', 'sum', and 'mean' options, where options should have an effect. NOTE: because of how data converts from event study to long (it only shifts period 2 (e.g. j2, y2) for the last row, as it assumes observations zigzag), it will only correct duplicates for period 1
    worker_data = []
    worker_data.append({'j1': 0, 'j2': 1, 't1': 1, 't2': 2, 'i': 0, 'y1': 2., 'y2': 1.})
    worker_data.append({'j1': 1, 'j2': 2, 't1': 1, 't2': 2, 'i': 1, 'y1': 0.5, 'y2': 1.5})
    worker_data.append({'j1': 1, 'j2': 2, 't1': 1, 't2': 2, 'i': 1, 'y1': 0.75, 'y2': 1.})
    worker_data.append({'j1': 2, 'j2': 1, 't1': 1, 't2': 2, 'i': 1, 'y1': 1., 'y2': 2.})
    worker_data.append({'j1': 2, 'j2': 2, 't1': 1, 't2': 1, 'i': 3, 'y1': 1., 'y2': 1.})
    worker_data.append({'j1': 2, 'j2': 2, 't1': 2, 't2': 2, 'i': 3, 'y1': 1., 'y2': 1.})
    worker_data.append({'j1': 1, 'j2': 1, 't1': 1, 't2': 1, 'i': 3, 'y1': 1.5, 'y2': 1.5})
    worker_data.append({'j1': 1, 'j2': 1, 't1': 2, 't2': 2, 'i': 3, 'y1': 1.5, 'y2': 1.5})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])[['i', 'j1', 'j2', 'y1', 'y2', 't1', 't2']]

    for how in ['max', 'sum', 'mean']:
        bdf = bpd.BipartiteEventStudy(data=df)
        bdf = bdf.clean_data({'i_t_how': how}).gen_m()

        stayers = bdf[bdf['m'] == 0]
        movers = bdf[bdf['m'] == 1]

        assert movers.iloc[0]['i'] == 0
        assert movers.iloc[0]['j1'] == 0
        assert movers.iloc[0]['j2'] == 1
        assert movers.iloc[0]['y1'] == 2
        assert movers.iloc[0]['y2'] == 1
        assert movers.iloc[0]['t1'] == 1
        assert movers.iloc[0]['t2'] == 2

        assert movers.iloc[1]['i'] == 1
        if how == 'max':
            assert movers.iloc[1]['j1'] == 2
            assert movers.iloc[1]['y1'] == 1
            assert movers.iloc[1]['j2'] == 1
            assert movers.iloc[1]['y2'] == 2
        elif how == 'sum':
            assert movers.iloc[1]['j1'] == 1
            assert movers.iloc[1]['y1'] == 1.25
            assert movers.iloc[1]['j2'] == 2
            assert movers.iloc[1]['y2'] == 2.5
        elif how == 'mean':
            assert movers.iloc[1]['j1'] == 2
            assert movers.iloc[1]['y1'] == 1
            assert movers.iloc[1]['j2'] == 1
            assert movers.iloc[1]['y2'] == 2
        assert movers.iloc[1]['t1'] == 1
        assert movers.iloc[1]['t2'] == 2

        assert stayers.iloc[0]['i'] == 2
        assert stayers.iloc[0]['j1'] == 1
        assert stayers.iloc[0]['j2'] == 1
        assert stayers.iloc[0]['y1'] == 1.5
        assert stayers.iloc[0]['y2'] == 1.5
        assert stayers.iloc[0]['t1'] == 1
        assert stayers.iloc[0]['t2'] == 1

        assert stayers.iloc[1]['i'] == 2
        assert stayers.iloc[1]['j1'] == 1
        assert stayers.iloc[1]['j2'] == 1
        assert stayers.iloc[1]['y1'] == 1.5
        assert stayers.iloc[1]['y2'] == 1.5
        assert stayers.iloc[1]['t1'] == 2
        assert stayers.iloc[1]['t2'] == 2

def test_string_ids_17():
    # String worker and firm ids.
    worker_data = []
    worker_data.append({'j': 'a', 't': 1, 'i': 'a', 'y': 2., 'index': 0})
    worker_data.append({'j': 'b', 't': 2, 'i': 'a', 'y': 1., 'index': 1})
    worker_data.append({'j': 'b', 't': 1, 'i': 'b', 'y': 1., 'index': 2})
    worker_data.append({'j': 'c', 't': 2, 'i': 'b', 'y': 1., 'index': 3})
    worker_data.append({'j': 'd', 't': 2, 'i': 'b', 'y': 0.5, 'index': 4})
    worker_data.append({'j': 'c', 't': 1, 'i': 'd', 'y': 1., 'index': 5})
    worker_data.append({'j': 'b', 't': 1, 'i': 'd', 'y': 1.5, 'index': 6})
    worker_data.append({'j': 'c', 't': 2, 'i': 'd', 'y': 1., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data().gen_m()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['y'] == 2
    assert movers.iloc[0]['t'] == 1

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['y'] == 1
    assert movers.iloc[1]['t'] == 2

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['y'] == 1
    assert movers.iloc[2]['t'] == 1

    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['j'] == 2
    assert movers.iloc[3]['y'] == 1
    assert movers.iloc[3]['t'] == 2

    assert movers.iloc[4]['i'] == 2
    assert movers.iloc[4]['j'] == 1
    assert movers.iloc[4]['y'] == 1.5
    assert movers.iloc[4]['t'] == 1

    assert movers.iloc[5]['i'] == 2
    assert movers.iloc[5]['j'] == 2
    assert movers.iloc[5]['y'] == 1
    assert movers.iloc[5]['t'] == 2

def test_general_methods_18():
    # Test some general methods, like n_workers/n_firms/n_clusters, included_cols(), drop(), and rename().
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'g': 2, 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'g': 1, 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'g': 1, 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'g': 2, 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 2, 'y': 1., 'g': 2, 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 2, 'y': 1., 'g': 2, 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't', 'g']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    assert bdf.n_workers() == 3
    assert bdf.n_firms() == 3
    assert bdf.n_clusters() == 2

    correct_cols = True
    all_cols = bdf._included_cols()
    for col in ['j', 't', 'i', 'y', 'g']:
        if col not in all_cols:
            correct_cols = False
            break
    assert correct_cols

    bdf.drop('g1')
    assert 'g1' in bdf.columns and 'g2' in bdf.columns

    bdf.drop('g')
    assert 'g1' not in bdf.columns and 'g2' not in bdf.columns

    bdf.rename({'i': 'w'})
    assert 'i' in bdf.columns

    bdf['g1'] = 1
    bdf['g2'] = 1
    bdf.col_dict['g1'] = 'g1'
    bdf.col_dict['g2'] = 'g2'
    assert 'g1' in bdf.columns and 'g2' in bdf.columns
    bdf.rename({'g': 'r'})
    assert 'g1' not in bdf.columns and 'g2' not in bdf.columns

def test_save_19():
    # Make sure changing attributes in a saved version does not overwrite values in the original.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 3, 't': 2, 'i': 1, 'y': 0.5, 'index': 4})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 5})
    worker_data.append({'j': 1, 't': 1, 'i': 3, 'y': 1.5, 'index': 6})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    # Long
    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf2 = bdf.copy()
    bdf2.gen_m()

    assert 'm' in bdf2._included_cols() and 'm' not in bdf._included_cols()

    # Event study
    bdf = bdf.get_es()
    bdf = bdf.clean_data().drop('m')
    bdf2 = bdf.copy()
    bdf2.gen_m()

    assert 'm' in bdf2._included_cols() and 'm' not in bdf._included_cols()

    # Collapsed long
    bdf = bdf.get_long().get_collapsed_long()
    bdf = bdf.clean_data().drop('m')
    bdf2 = bdf.copy()
    bdf2.gen_m()

    assert 'm' in bdf2._included_cols() and 'm' not in bdf._included_cols()

    # Collapsed event study
    bdf = bdf.get_es()
    bdf = bdf.clean_data().drop('m')
    bdf2 = bdf.copy()
    bdf2.gen_m()

    assert 'm' in bdf2._included_cols() and 'm' not in bdf._included_cols()

def test_id_reference_dict_20():
    # String worker and firm ids, link with id_reference_dict.
    worker_data = []
    worker_data.append({'j': 'a', 't': 1, 'i': 'a', 'y': 2., 'index': 0})
    worker_data.append({'j': 'b', 't': 2, 'i': 'a', 'y': 1., 'index': 1})
    worker_data.append({'j': 'b', 't': 1, 'i': 'b', 'y': 1., 'index': 2})
    worker_data.append({'j': 'c', 't': 2, 'i': 'b', 'y': 1., 'index': 3})
    worker_data.append({'j': 'd', 't': 2, 'i': 'b', 'y': 0.5, 'index': 4})
    worker_data.append({'j': 'c', 't': 1, 'i': 'd', 'y': 1., 'index': 5})
    worker_data.append({'j': 'b', 't': 1, 'i': 'd', 'y': 1.5, 'index': 6})
    worker_data.append({'j': 'c', 't': 2, 'i': 'd', 'y': 1., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True)
    bdf = bdf.clean_data().gen_m()

    id_reference_dict = bdf.id_reference_dict

    merge_df = bdf.merge(id_reference_dict['i'], how='left', left_on='i', right_on='adjusted_ids_1').rename({'original_ids': 'original_i'})
    merge_df = merge_df.merge(id_reference_dict['j'], how='left', left_on='j', right_on='adjusted_ids_1').rename({'original_ids': 'original_j'})

    stayers = merge_df[merge_df['m'] == 0]
    movers = merge_df[merge_df['m'] == 1]

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['original_i'] == 'a'
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['original_j'] == 'a'
    assert movers.iloc[0]['y'] == 2
    assert movers.iloc[0]['t'] == 1

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['original_i'] == 'a'
    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['original_j'] == 'b'
    assert movers.iloc[1]['y'] == 1
    assert movers.iloc[1]['t'] == 2

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['original_i'] == 'b'
    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['original_j'] == 'b'
    assert movers.iloc[2]['y'] == 1
    assert movers.iloc[2]['t'] == 1

    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['original_i'] == 'b'
    assert movers.iloc[3]['j'] == 2
    assert movers.iloc[3]['original_j'] == 'c'
    assert movers.iloc[3]['y'] == 1
    assert movers.iloc[3]['t'] == 2

    assert movers.iloc[4]['i'] == 2
    assert movers.iloc[4]['original_i'] == 'd'
    assert movers.iloc[4]['j'] == 1
    assert movers.iloc[4]['original_j'] == 'b'
    assert movers.iloc[4]['y'] == 1.5
    assert movers.iloc[4]['t'] == 1

    assert movers.iloc[5]['i'] == 2
    assert movers.iloc[5]['original_i'] == 'd'
    assert movers.iloc[5]['j'] == 2
    assert movers.iloc[5]['original_j'] == 'c'
    assert movers.iloc[5]['y'] == 1
    assert movers.iloc[5]['t'] == 2

def test_id_reference_dict_22():
    # String worker and firm ids, link with id_reference_dict. Testing original_ids() method.
    worker_data = []
    worker_data.append({'j': 'a', 't': 1, 'i': 'a', 'y': 2., 'index': 0})
    worker_data.append({'j': 'b', 't': 2, 'i': 'a', 'y': 1., 'index': 1})
    worker_data.append({'j': 'b', 't': 1, 'i': 'b', 'y': 1., 'index': 2})
    worker_data.append({'j': 'c', 't': 2, 'i': 'b', 'y': 1., 'index': 3})
    worker_data.append({'j': 'd', 't': 2, 'i': 'b', 'y': 0.5, 'index': 4})
    worker_data.append({'j': 'c', 't': 1, 'i': 'd', 'y': 1., 'index': 5})
    worker_data.append({'j': 'b', 't': 1, 'i': 'd', 'y': 1.5, 'index': 6})
    worker_data.append({'j': 'c', 't': 2, 'i': 'd', 'y': 1., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True)
    bdf = bdf.clean_data().gen_m()

    merge_df = bdf.original_ids()

    stayers = merge_df[merge_df['m'] == 0]
    movers = merge_df[merge_df['m'] == 1]

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['original_i'] == 'a'
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['original_j'] == 'a'
    assert movers.iloc[0]['y'] == 2
    assert movers.iloc[0]['t'] == 1

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['original_i'] == 'a'
    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['original_j'] == 'b'
    assert movers.iloc[1]['y'] == 1
    assert movers.iloc[1]['t'] == 2

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['original_i'] == 'b'
    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['original_j'] == 'b'
    assert movers.iloc[2]['y'] == 1
    assert movers.iloc[2]['t'] == 1

    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['original_i'] == 'b'
    assert movers.iloc[3]['j'] == 2
    assert movers.iloc[3]['original_j'] == 'c'
    assert movers.iloc[3]['y'] == 1
    assert movers.iloc[3]['t'] == 2

    assert movers.iloc[4]['i'] == 2
    assert movers.iloc[4]['original_i'] == 'd'
    assert movers.iloc[4]['j'] == 1
    assert movers.iloc[4]['original_j'] == 'b'
    assert movers.iloc[4]['y'] == 1.5
    assert movers.iloc[4]['t'] == 1

    assert movers.iloc[5]['i'] == 2
    assert movers.iloc[5]['original_i'] == 'd'
    assert movers.iloc[5]['j'] == 2
    assert movers.iloc[5]['original_j'] == 'c'
    assert movers.iloc[5]['y'] == 1
    assert movers.iloc[5]['t'] == 2

def test_id_reference_dict_23():
    # String worker and firm ids, link with id_reference_dict. Testing original_ids() method where there are multiple steps of references.
    worker_data = []
    worker_data.append({'j': 'a', 't': 1, 'i': 'a', 'y': 2., 'index': 0})
    worker_data.append({'j': 'b', 't': 2, 'i': 'a', 'y': 1., 'index': 1})
    worker_data.append({'j': 'b', 't': 1, 'i': 'b', 'y': 1., 'index': 2})
    worker_data.append({'j': 'c', 't': 2, 'i': 'b', 'y': 1., 'index': 3})
    worker_data.append({'j': 'd', 't': 2, 'i': 'b', 'y': 0.5, 'index': 4})
    worker_data.append({'j': 'c', 't': 1, 'i': 'd', 'y': 1., 'index': 5})
    worker_data.append({'j': 'b', 't': 1, 'i': 'd', 'y': 1.5, 'index': 6})
    worker_data.append({'j': 'c', 't': 2, 'i': 'd', 'y': 1., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True)
    bdf = bdf.clean_data().gen_m()
    bdf = bdf[bdf['j'] == 1]
    bdf = bdf.clean_data()

    merge_df = bdf.original_ids()

    stayers = merge_df[merge_df['m'] == 0]
    movers = merge_df[merge_df['m'] == 1]

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['original_i'] == 'a'
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['original_j'] == 'b'
    assert movers.iloc[0]['y'] == 1
    assert movers.iloc[0]['t'] == 2

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['original_i'] == 'b'
    assert movers.iloc[1]['j'] == 0
    assert movers.iloc[1]['original_j'] == 'b'
    assert movers.iloc[1]['y'] == 1
    assert movers.iloc[1]['t'] == 1

    assert movers.iloc[2]['i'] == 2
    assert movers.iloc[2]['original_i'] == 'd'
    assert movers.iloc[2]['j'] == 0
    assert movers.iloc[2]['original_j'] == 'b'
    assert movers.iloc[2]['y'] == 1.5
    assert movers.iloc[2]['t'] == 1

def test_fill_time_24_1():
    # Test .fill_time() method for long format, with no data to fill in.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data().gen_m()
    new_df = bdf.fill_periods()

    stayers = new_df[new_df['m'] == 0]
    movers = new_df[new_df['m'] == 1]

    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y'] == 2

    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['y'] == 1

    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['y'] == 1

    assert movers.iloc[3]['j'] == 2
    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['y'] == 1

    assert stayers.iloc[0]['j'] == 2
    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['y'] == 1

def test_fill_time_24_2():
    # Test .fill_time() method for long format, with 1 row of data to fill in.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 3, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data().gen_m()
    new_df = bdf.fill_periods()

    stayers = new_df[new_df['m'] == 0]
    movers = new_df[new_df['m'] == 1]

    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y'] == 2

    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['y'] == 1

    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['y'] == 1

    assert movers.iloc[3]['j'] == - 1
    assert movers.iloc[3]['i'] == 1
    assert np.isnan(movers.iloc[3]['y'])

    assert movers.iloc[4]['j'] == 2
    assert movers.iloc[4]['i'] == 1
    assert movers.iloc[4]['y'] == 1

    assert stayers.iloc[0]['j'] == 2
    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['y'] == 1

def test_fill_time_24_3():
    # Test .fill_time() method for long format, with 2 rows of data to fill in.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 4, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data().gen_m()
    new_df = bdf.fill_periods()

    stayers = new_df[new_df['m'] == 0]
    movers = new_df[new_df['m'] == 1]

    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y'] == 2

    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['y'] == 1

    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['y'] == 1

    assert movers.iloc[3]['j'] == - 1
    assert movers.iloc[3]['i'] == 1
    assert np.isnan(movers.iloc[3]['y'])
    assert movers.iloc[3]['m'] == 1

    assert movers.iloc[4]['j'] == - 1
    assert movers.iloc[4]['i'] == 1
    assert np.isnan(movers.iloc[4]['y'])
    assert movers.iloc[4]['m'] == 1

    assert movers.iloc[5]['j'] == 2
    assert movers.iloc[5]['i'] == 1
    assert movers.iloc[5]['y'] == 1

    assert stayers.iloc[0]['j'] == 2
    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['y'] == 1

def test_uncollapse_25():
    # Convert from collapsed long to long format.
    worker_data = []
    worker_data.append({'j': 0, 't1': 1, 't2': 1, 'i': 0, 'y': 2.})
    worker_data.append({'j': 1, 't1': 2, 't2': 2, 'i': 0, 'y': 1.})
    worker_data.append({'j': 1, 't1': 1, 't2': 2, 'i': 1, 'y': 1.})
    worker_data.append({'j': 2, 't1': 2, 't2': 2, 'i': 1, 'y': 1.})
    worker_data.append({'j': 2, 't1': 2, 't2': 2, 'i': 1, 'y': 1.5})
    worker_data.append({'j': 3, 't1': 2, 't2': 2, 'i': 1, 'y': 0.5})
    worker_data.append({'j': 2, 't1': 1, 't2': 2, 'i': 3, 'y': 1.})
    worker_data.append({'j': 1, 't1': 1, 't2': 2, 'i': 3, 'y': 1.5})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])[['i', 'j', 'y', 't1', 't2']]

    bdf = bpd.BipartiteLongCollapsed(data=df).uncollapse()

    assert bdf.iloc[0]['i'] == 0
    assert bdf.iloc[0]['j'] == 0
    assert bdf.iloc[0]['y'] == 2
    assert bdf.iloc[0]['t'] == 1

    assert bdf.iloc[1]['i'] == 0
    assert bdf.iloc[1]['j'] == 1
    assert bdf.iloc[1]['y'] == 1
    assert bdf.iloc[1]['t'] == 2

    assert bdf.iloc[2]['i'] == 1
    assert bdf.iloc[2]['j'] == 1
    assert bdf.iloc[2]['y'] == 1
    assert bdf.iloc[2]['t'] == 1

    assert bdf.iloc[3]['i'] == 1
    assert bdf.iloc[3]['j'] == 1
    assert bdf.iloc[3]['y'] == 1
    assert bdf.iloc[3]['t'] == 2

    assert bdf.iloc[4]['i'] == 1
    assert bdf.iloc[4]['j'] == 2
    assert bdf.iloc[4]['y'] == 1
    assert bdf.iloc[4]['t'] == 2

    assert bdf.iloc[5]['i'] == 1
    assert bdf.iloc[5]['j'] == 2
    assert bdf.iloc[5]['y'] == 1.5
    assert bdf.iloc[5]['t'] == 2

    assert bdf.iloc[6]['i'] == 1
    assert bdf.iloc[6]['j'] == 3
    assert bdf.iloc[6]['y'] == 0.5
    assert bdf.iloc[6]['t'] == 2

    assert bdf.iloc[7]['i'] == 3
    assert bdf.iloc[7]['j'] == 2
    assert bdf.iloc[7]['y'] == 1
    assert bdf.iloc[7]['t'] == 1

    assert bdf.iloc[8]['i'] == 3
    assert bdf.iloc[8]['j'] == 2
    assert bdf.iloc[8]['y'] == 1
    assert bdf.iloc[8]['t'] == 2

    assert bdf.iloc[9]['i'] == 3
    assert bdf.iloc[9]['j'] == 1
    assert bdf.iloc[9]['y'] == 1.5
    assert bdf.iloc[9]['t'] == 1

    assert bdf.iloc[10]['i'] == 3
    assert bdf.iloc[10]['j'] == 1
    assert bdf.iloc[10]['y'] == 1.5
    assert bdf.iloc[10]['t'] == 2

###################################
##### Tests for BipartiteLong #####
###################################

def test_long_get_es_extended_1():
    # Test get_es_extended() by making sure it is generating the event study correctly for periods_pre=2 and periods_post=1
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2.})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1.})
    worker_data.append({'j': 1, 't': 3, 'i': 0, 'y': 1.})
    worker_data.append({'j': 0, 't': 4, 'i': 0, 'y': 1.})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1.})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1.})
    worker_data.append({'j': 2, 't': 3, 'i': 1, 'y': 2.})
    worker_data.append({'j': 5, 't': 3, 'i': 1, 'y': 1.})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1.})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1.})
    worker_data.append({'j': 3, 't': 3, 'i': 3, 'y': 1.5})
    worker_data.append({'j': 0, 't': 1, 'i': 4, 'y': 1.})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])[['i', 'j', 'y', 't']]
    df['g'] = df['j'] # Fill in g column as j

    bdf = bpd.BipartiteLong(df)
    bdf = bdf.clean_data()

    es_extended = bdf.get_es_extended(periods_pre=2, periods_post=1)

    assert es_extended.iloc[0]['i'] == 0
    assert es_extended.iloc[0]['t'] == 4
    assert es_extended.iloc[0]['g_l2'] == 1
    assert es_extended.iloc[0]['g_l1'] == 1
    assert es_extended.iloc[0]['g_f1'] == 0
    assert es_extended.iloc[0]['y_l2'] == 1
    assert es_extended.iloc[0]['y_l1'] == 1
    assert es_extended.iloc[0]['y_f1'] == 1

    assert es_extended.iloc[1]['i'] == 2
    assert es_extended.iloc[1]['t'] == 3
    assert es_extended.iloc[1]['g_l2'] == 2
    assert es_extended.iloc[1]['g_l1'] == 2
    assert es_extended.iloc[1]['g_f1'] == 3
    assert es_extended.iloc[1]['y_l2'] == 1
    assert es_extended.iloc[1]['y_l1'] == 1
    assert es_extended.iloc[1]['y_f1'] == 1.5

def test_long_get_es_extended_2():
    # Test get_es_extended() by making sure workers move firms at the fulcrum of the event study
    sim_data = bpd.SimBipartite().sim_network()
    sim_data['g'] = sim_data['j'] # Fill in g column as j
    bdf = bpd.BipartiteLong(sim_data)
    bdf = bdf.clean_data()

    es_extended = bdf.get_es_extended(periods_pre=3, periods_post=2)

    assert np.sum(es_extended['g_l1'] == es_extended['g_f1']) == 0

def test_long_get_es_extended_3_1():
    # Test get_es_extended() by making sure workers move firms at the fulcrum of the event study and stable_pre works
    sim_data = bpd.SimBipartite().sim_network()
    sim_data['g'] = sim_data['j'] # Fill in g column as j
    bdf = bpd.BipartiteLong(sim_data)
    bdf = bdf.clean_data()

    es_extended = bdf.get_es_extended(periods_pre=2, periods_post=3, stable_pre=True)

    assert np.sum(es_extended['g_l1'] == es_extended['g_f1']) == 0
    assert np.sum(es_extended['g_l2'] != es_extended['g_l1']) == 0

def test_long_get_es_extended_3_2():
    # Test get_es_extended() by making sure workers move firms at the fulcrum of the event study and stable_post works
    sim_data = bpd.SimBipartite().sim_network()
    sim_data['g'] = sim_data['j'] # Fill in g column as j
    bdf = bpd.BipartiteLong(sim_data)
    bdf = bdf.clean_data()

    es_extended = bdf.get_es_extended(periods_pre=3, periods_post=2, stable_post=True)

    assert np.sum(es_extended['g_l1'] == es_extended['g_f1']) == 0
    assert np.sum(es_extended['g_f1'] != es_extended['g_f2']) == 0

############################################
##### Tests for BipartiteLongCollapsed #####
############################################

def test_long_collapsed_1():
    # Test constructor for BipartiteLongCollapsed.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    df = pd.DataFrame(bdf.get_collapsed_long()).rename({'y': 'y'}, axis=1)
    bdf = bpd.BipartiteLongCollapsed(df, col_dict={'y': 'y'})
    bdf = bdf.clean_data()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['y'] == 2

    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['y'] == 1

    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['y'] == 1

    assert movers.iloc[3]['j'] == 2
    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['y'] == 1

    assert stayers.iloc[0]['j'] == 2
    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['y'] == 1

#########################################
##### Tests for BipartiteEventStudy #####
#########################################

def test_event_study_1():
    # Test constructor for BipartiteEventStudy.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    df = pd.DataFrame(bdf.get_es()).rename({'t1': 't'}, axis=1)
    bdf = bpd.BipartiteEventStudy(df, col_dict={'t1': 't'})
    bdf = bdf.clean_data()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1
    assert movers.iloc[0]['t1'] == 1
    assert movers.iloc[0]['t2'] == 2

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1
    assert movers.iloc[1]['t1'] == 1
    assert movers.iloc[1]['t2'] == 2

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['t1'] == 1
    assert stayers.iloc[0]['t2'] == 1

    assert stayers.iloc[1]['i'] == 2
    assert stayers.iloc[1]['j1'] == 2
    assert stayers.iloc[1]['j2'] == 2
    assert stayers.iloc[1]['y1'] == 2
    assert stayers.iloc[1]['y2'] == 2
    assert stayers.iloc[1]['t1'] == 2
    assert stayers.iloc[1]['t2'] == 2

def test_get_cs_2():
    # Test get_cs() for BipartiteEventStudy.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 4})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    df = pd.DataFrame(bdf.get_es()).rename({'t1': 't'}, axis=1)
    bdf = bpd.BipartiteEventStudy(df, col_dict={'t1': 't'})
    bdf = bdf.clean_data()
    bdf = bdf.get_cs()

    stayers = bdf[bdf['m'] == 0]
    movers1 = bdf[(bdf['m'] == 1) & (bdf['cs'] == 1)]
    movers0 = bdf[(bdf['m'] == 1) & (bdf['cs'] == 0)]

    assert movers1.iloc[0]['i'] == 0
    assert movers1.iloc[0]['j1'] == 0
    assert movers1.iloc[0]['j2'] == 1
    assert movers1.iloc[0]['y1'] == 2
    assert movers1.iloc[0]['y2'] == 1
    assert movers1.iloc[0]['t1'] == 1
    assert movers1.iloc[0]['t2'] == 2

    assert movers1.iloc[1]['i'] == 1
    assert movers1.iloc[1]['j1'] == 1
    assert movers1.iloc[1]['j2'] == 2
    assert movers1.iloc[1]['y1'] == 1
    assert movers1.iloc[1]['y2'] == 1
    assert movers1.iloc[1]['t1'] == 1
    assert movers1.iloc[1]['t2'] == 2

    assert movers0.iloc[0]['i'] == 0
    assert movers0.iloc[0]['j1'] == 1
    assert movers0.iloc[0]['j2'] == 0
    assert movers0.iloc[0]['y1'] == 1
    assert movers0.iloc[0]['y2'] == 2
    assert movers0.iloc[0]['t1'] == 2
    assert movers0.iloc[0]['t2'] == 1

    assert movers0.iloc[1]['i'] == 1
    assert movers0.iloc[1]['j1'] == 2
    assert movers0.iloc[1]['j2'] == 1
    assert movers0.iloc[1]['y1'] == 1
    assert movers0.iloc[1]['y2'] == 1
    assert movers0.iloc[1]['t1'] == 2
    assert movers0.iloc[1]['t2'] == 1

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['t1'] == 1
    assert stayers.iloc[0]['t2'] == 1

    assert stayers.iloc[1]['i'] == 2
    assert stayers.iloc[1]['j1'] == 2
    assert stayers.iloc[1]['j2'] == 2
    assert stayers.iloc[1]['y1'] == 2
    assert stayers.iloc[1]['y2'] == 2
    assert stayers.iloc[1]['t1'] == 2
    assert stayers.iloc[1]['t2'] == 2

##################################################
##### Tests for BipartiteEventStudyCollapsed #####
##################################################

def test_event_study_collapsed_1():
    # Test constructor for BipartiteEventStudyCollapsed.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 3, 'i': 1, 'y': 2., 'index': 4})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 5})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'index': 6})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    df = pd.DataFrame(bdf.get_es()).rename({'y1': 'comp1'}, axis=1)
    bdf = bpd.BipartiteEventStudyCollapsed(df, col_dict={'y1': 'comp1'})
    bdf = bdf.clean_data()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1
    assert movers.iloc[0]['t11'] == 1
    assert movers.iloc[0]['t12'] == 1
    assert movers.iloc[0]['t21'] == 2
    assert movers.iloc[0]['t22'] == 2

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1.5
    assert movers.iloc[1]['t11'] == 1
    assert movers.iloc[1]['t12'] == 1
    assert movers.iloc[1]['t21'] == 2
    assert movers.iloc[1]['t22'] == 3

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['t11'] == 1
    assert stayers.iloc[0]['t12'] == 2
    assert stayers.iloc[0]['t21'] == 1
    assert stayers.iloc[0]['t22'] == 2

def test_get_cs_2():
    # Test get_cs() for BipartiteEventStudyCollapsed.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'index': 0})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'index': 1})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'index': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'index': 3})
    worker_data.append({'j': 2, 't': 3, 'i': 1, 'y': 2., 'index': 4})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'index': 5})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'index': 6})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['i', 'j', 'y', 't']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    df = pd.DataFrame(bdf.get_es()).rename({'y1': 'comp1'}, axis=1)
    bdf = bpd.BipartiteEventStudyCollapsed(df, col_dict={'y1': 'comp1'})
    bdf = bdf.clean_data()
    bdf = bdf.get_cs()

    stayers = bdf[bdf['m'] == 0]
    movers1 = bdf[(bdf['m'] == 1) & (bdf['cs'] == 1)]
    movers0 = bdf[(bdf['m'] == 1) & (bdf['cs'] == 0)]

    assert movers1.iloc[0]['i'] == 0
    assert movers1.iloc[0]['j1'] == 0
    assert movers1.iloc[0]['j2'] == 1
    assert movers1.iloc[0]['y1'] == 2
    assert movers1.iloc[0]['y2'] == 1
    assert movers1.iloc[0]['t11'] == 1
    assert movers1.iloc[0]['t12'] == 1
    assert movers1.iloc[0]['t21'] == 2
    assert movers1.iloc[0]['t22'] == 2

    assert movers1.iloc[1]['i'] == 1
    assert movers1.iloc[1]['j1'] == 1
    assert movers1.iloc[1]['j2'] == 2
    assert movers1.iloc[1]['y1'] == 1
    assert movers1.iloc[1]['y2'] == 1.5
    assert movers1.iloc[1]['t11'] == 1
    assert movers1.iloc[1]['t12'] == 1
    assert movers1.iloc[1]['t21'] == 2
    assert movers1.iloc[1]['t22'] == 3

    assert movers0.iloc[0]['i'] == 0
    assert movers0.iloc[0]['j1'] == 1
    assert movers0.iloc[0]['j2'] == 0
    assert movers0.iloc[0]['y1'] == 1
    assert movers0.iloc[0]['y2'] == 2
    assert movers0.iloc[0]['t11'] == 2
    assert movers0.iloc[0]['t12'] == 2
    assert movers0.iloc[0]['t21'] == 1
    assert movers0.iloc[0]['t22'] == 1

    assert movers0.iloc[1]['i'] == 1
    assert movers0.iloc[1]['j1'] == 2
    assert movers0.iloc[1]['j2'] == 1
    assert movers0.iloc[1]['y1'] == 1.5
    assert movers0.iloc[1]['y2'] == 1
    assert movers0.iloc[1]['t11'] == 2
    assert movers0.iloc[1]['t12'] == 3
    assert movers0.iloc[1]['t21'] == 1
    assert movers0.iloc[1]['t22'] == 1

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['t11'] == 1
    assert stayers.iloc[0]['t12'] == 2
    assert stayers.iloc[0]['t21'] == 1
    assert stayers.iloc[0]['t22'] == 2

#####################################
##### Tests for BipartitePandas #####
#####################################

def test_reformatting_1():
    # Convert from long --> event study --> long --> collapsed long --> collapsed event study --> collapsed long to ensure conversion maintains data properly.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'g': 1})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'g': 2})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'g': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'g': 1})
    worker_data.append({'j': 2, 't': 3, 'i': 1, 'y': 2., 'g': 1})
    worker_data.append({'j': 5, 't': 3, 'i': 1, 'y': 1., 'g': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'g': 1})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'g': 1})
    worker_data.append({'j': 0, 't': 1, 'i': 4, 'y': 1., 'g': 1})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])[['i', 'j', 'y', 't', 'g']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_es()
    bdf = bdf.clean_data()
    bdf = bdf.get_long()
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.clean_data()
    bdf = bdf.get_es()
    bdf = bdf.clean_data()
    bdf = bdf.get_long()
    bdf = bdf.clean_data()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

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

# ################################
# ##### Tests for Clustering #####
# ################################

def test_cluster_1():
    # Test cluster function is working correctly for long format.
    nk = 10
    sim_data = bpd.SimBipartite({'nk': nk}).sim_network()
    bdf = bpd.BipartiteLong(sim_data)
    bdf = bdf.clean_data()

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.cdfs(measure=measure)
            grouping = bpd.grouping.kmeans(n_clusters=nk)
            bdf = bdf.cluster(measures=measures, grouping=grouping, stayers_movers=stayers_movers)

            clusters_true = sim_data[~bdf['g'].isna()]['psi'].astype('category').cat.codes.astype(int).to_numpy() # Skip firms that aren't clustered
            clusters_estimated = bdf[~bdf['g'].isna()]['g'].astype(int).to_numpy() # Skip firms that aren't clustered

            # Find which clusters are most often matched together
            replace_df = pd.DataFrame({'psi': clusters_true, 'psi_est': clusters_estimated}, index=np.arange(len(clusters_true)))
            clusters_available = list(np.arange(nk)) # Which clusters have yet to be used
            matches_available = list(np.arange(nk)) # Which matches have yet to be used
            clusters_match = []
            matches_match = []
            # Iterate through clusters to find matches, but ensure no duplicate matches
            for i in range(nk):
                best_proportion = - 1 # Best proportion of matches
                best_cluster = None # Best cluster
                best_match = None # Best match
                for g in clusters_available: # Iterate over remaining clusters
                    cluster_df = replace_df[replace_df['psi'] == g]
                    value_counts = cluster_df[cluster_df['psi_est'].isin(matches_available)].value_counts() # Only show valid remaining matches
                    if len(value_counts) > 0:
                        proportion = value_counts.iloc[0] / len(cluster_df)
                    else:
                        proportion = 0
                    if proportion > best_proportion:
                        best_proportion = proportion
                        best_cluster = g
                        if len(value_counts) > 0:
                            best_match = value_counts.index[0][1]
                        else:
                            best_match = matches_available[0] # Just take a random cluster
                # Use best cluster
                clusters_match.append(best_cluster)
                matches_match.append(best_match)
                del clusters_available[clusters_available.index(best_cluster)]
                del matches_available[matches_available.index(best_match)]
            match_df = pd.DataFrame({'psi': clusters_match, 'psi_est': matches_match}, index=np.arange(nk))
            # replace_df = replace_df.groupby('psi').apply(lambda a: a.value_counts().index[0][1])

            clusters_merged = pd.merge(pd.DataFrame({'psi': clusters_true}), match_df, how='left', on='psi')

            wrong_cluster = np.sum(clusters_merged['psi_est'] != clusters_estimated)
            if measure == 'quantile_all':
                bound = 5000 # 10% error
            elif measure == 'quantile_firm_small':
                bound = 10000 # 20% error
            elif measure == 'quantile_firm_large':
                bound = 10000 # 20% error
            if stayers_movers == 'stayers':
                bound = 35000 # 70% error

            assert wrong_cluster < bound, 'error is {} for {}'.format(wrong_cluster, measure)

def test_cluster_2():
    # Test cluster function is working correctly for event study format.
    nk = 10
    sim_data = bpd.SimBipartite({'nk': nk}).sim_network()
    bdf = bpd.BipartiteLong(sim_data)
    bdf = bdf.clean_data()
    bdf = bdf.get_es()

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.cdfs(measure=measure)
            grouping = bpd.grouping.kmeans(n_clusters=nk)
            bdf = bdf.cluster(measures=measures, grouping=grouping, stayers_movers=stayers_movers)

            to_long = bdf.get_long()
            clusters_true = sim_data[~to_long['g'].isna()]['psi'].astype('category').cat.codes.astype(int).to_numpy() # Skip firms that aren't clustered
            clusters_estimated = to_long[~to_long['g'].isna()]['g'].astype(int).to_numpy() # Skip firms that aren't clustered

            # Find which clusters are most often matched together
            replace_df = pd.DataFrame({'psi': clusters_true, 'psi_est': clusters_estimated}, index=np.arange(len(clusters_true)))
            clusters_available = list(np.arange(nk)) # Which clusters have yet to be used
            matches_available = list(np.arange(nk)) # Which matches have yet to be used
            clusters_match = []
            matches_match = []
            # Iterate through clusters to find matches, but ensure no duplicate matches
            for i in range(nk):
                best_proportion = - 1 # Best proportion of matches
                best_cluster = None # Best cluster
                best_match = None # Best match
                for g in clusters_available: # Iterate over remaining clusters
                    cluster_df = replace_df[replace_df['psi'] == g]
                    value_counts = cluster_df[cluster_df['psi_est'].isin(matches_available)].value_counts() # Only show valid remaining matches
                    if len(value_counts) > 0:
                        proportion = value_counts.iloc[0] / len(cluster_df)
                    else:
                        proportion = 0
                    if proportion > best_proportion:
                        best_proportion = proportion
                        best_cluster = g
                        if len(value_counts) > 0:
                            best_match = value_counts.index[0][1]
                        else:
                            best_match = matches_available[0] # Just take a random cluster
                # Use best cluster
                clusters_match.append(best_cluster)
                matches_match.append(best_match)
                del clusters_available[clusters_available.index(best_cluster)]
                del matches_available[matches_available.index(best_match)]
            match_df = pd.DataFrame({'psi': clusters_match, 'psi_est': matches_match}, index=np.arange(nk))
            # replace_df = replace_df.groupby('psi').apply(lambda a: a.value_counts().index[0][1])

            clusters_merged = pd.merge(pd.DataFrame({'psi': clusters_true}), match_df, how='left', on='psi')

            wrong_cluster = np.sum(clusters_merged['psi_est'] != clusters_estimated)
            if measure == 'quantile_all':
                bound = 5000 # 10% error
            elif measure == 'quantile_firm_small':
                bound = 10000 # 20% error
            elif measure == 'quantile_firm_large':
                bound = 10500 # 21% error
            if stayers_movers == 'stayers':
                bound = 35000 # 70% error

            assert wrong_cluster < bound, 'error is {} for {}'.format(wrong_cluster, measure)

def test_cluster_3():
    # Test cluster function is working correctly for collapsed long format.
    nk = 10
    sim_data = bpd.SimBipartite({'nk': nk}).sim_network()
    bdf = bpd.BipartiteLong(sim_data)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    # Compute spells
    sim_data['i_l1'] = sim_data['i'].shift(1)
    sim_data['j_l1'] = sim_data['j'].shift(1)
    sim_data['new_spell'] = (sim_data['i'] != sim_data['i_l1']) | (sim_data['j'] != sim_data['j_l1'])
    sim_data['spell'] = sim_data['new_spell'].cumsum()
    sim_data_spell = sim_data.groupby('spell').first()

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.cdfs(measure=measure)
            grouping = bpd.grouping.kmeans(n_clusters=nk)
            bdf = bdf.cluster(measures=measures, grouping=grouping, stayers_movers=stayers_movers)
            remaining_jids = bdf.dropna()['j'].unique()

            clusters_true = sim_data_spell[sim_data_spell['j'].isin(remaining_jids)]['psi'].astype('category').cat.codes.astype(int).to_numpy() # Skip firms that aren't clustered
            clusters_estimated = bdf[~bdf['g'].isna()]['g'].astype(int).to_numpy() # Skip firms that aren't clustered

            # Find which clusters are most often matched together
            replace_df = pd.DataFrame({'psi': clusters_true, 'psi_est': clusters_estimated}, index=np.arange(len(clusters_true)))
            clusters_available = list(np.arange(nk)) # Which clusters have yet to be used
            matches_available = list(np.arange(nk)) # Which matches have yet to be used
            clusters_match = []
            matches_match = []
            # Iterate through clusters to find matches, but ensure no duplicate matches
            for i in range(nk):
                best_proportion = - 1 # Best proportion of matches
                best_cluster = None # Best cluster
                best_match = None # Best match
                for g in clusters_available: # Iterate over remaining clusters
                    cluster_df = replace_df[replace_df['psi'] == g]
                    value_counts = cluster_df[cluster_df['psi_est'].isin(matches_available)].value_counts() # Only show valid remaining matches
                    if len(value_counts) > 0:
                        proportion = value_counts.iloc[0] / len(cluster_df)
                    else:
                        proportion = 0
                    if proportion > best_proportion:
                        best_proportion = proportion
                        best_cluster = g
                        if len(value_counts) > 0:
                            best_match = value_counts.index[0][1]
                        else:
                            best_match = matches_available[0] # Just take a random cluster
                # Use best cluster
                clusters_match.append(best_cluster)
                matches_match.append(best_match)
                del clusters_available[clusters_available.index(best_cluster)]
                del matches_available[matches_available.index(best_match)]
            match_df = pd.DataFrame({'psi': clusters_match, 'psi_est': matches_match}, index=np.arange(nk))
            # replace_df = replace_df.groupby('psi').apply(lambda a: a.value_counts().index[0][1])

            clusters_merged = pd.merge(pd.DataFrame({'psi': clusters_true}), match_df, how='left', on='psi')

            wrong_cluster = np.sum(clusters_merged['psi_est'] != clusters_estimated)
            if measure == 'quantile_all':
                bound = 5000 # 10% error
            elif measure == 'quantile_firm_small':
                bound = 10000 # 20% error
            elif measure == 'quantile_firm_large':
                bound = 10000 # 20% error
            if stayers_movers == 'stayers':
                bound = 35000 # 70% error

            assert wrong_cluster < bound, 'error is {} for {}'.format(wrong_cluster, measure)

def test_cluster_4():
    # Test cluster function is working correctly for collapsed event study format.
    nk = 10
    sim_data = bpd.SimBipartite({'nk': nk}).sim_network()
    bdf = bpd.BipartiteLong(sim_data)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long().get_es()
    # Compute spells
    sim_data['i_l1'] = sim_data['i'].shift(1)
    sim_data['j_l1'] = sim_data['j'].shift(1)
    sim_data['new_spell'] = (sim_data['i'] != sim_data['i_l1']) | (sim_data['j'] != sim_data['j_l1'])
    sim_data['spell'] = sim_data['new_spell'].cumsum()
    sim_data_spell = sim_data.groupby('spell').first()

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.cdfs(measure=measure)
            grouping = bpd.grouping.kmeans(n_clusters=nk)
            bdf = bdf.cluster(measures=measures, grouping=grouping, stayers_movers=stayers_movers)

            to_collapsed_long = bdf.get_long()
            remaining_jids = to_collapsed_long.dropna()['j'].unique()

            clusters_true = sim_data_spell[sim_data_spell['j'].isin(remaining_jids)]['psi'].astype('category').cat.codes.astype(int).to_numpy() # Skip firms that aren't clustered
            clusters_estimated = to_collapsed_long[~to_collapsed_long['g'].isna()]['g'].astype(int).to_numpy() # Skip firms that aren't clustered

            # Find which clusters are most often matched together
            replace_df = pd.DataFrame({'psi': clusters_true, 'psi_est': clusters_estimated}, index=np.arange(len(clusters_true)))
            clusters_available = list(np.arange(nk)) # Which clusters have yet to be used
            matches_available = list(np.arange(nk)) # Which matches have yet to be used
            clusters_match = []
            matches_match = []
            # Iterate through clusters to find matches, but ensure no duplicate matches
            for i in range(nk):
                best_proportion = - 1 # Best proportion of matches
                best_cluster = None # Best cluster
                best_match = None # Best match
                for g in clusters_available: # Iterate over remaining clusters
                    cluster_df = replace_df[replace_df['psi'] == g]
                    value_counts = cluster_df[cluster_df['psi_est'].isin(matches_available)].value_counts() # Only show valid remaining matches
                    if len(value_counts) > 0:
                        proportion = value_counts.iloc[0] / len(cluster_df)
                    else:
                        proportion = 0
                    if proportion > best_proportion:
                        best_proportion = proportion
                        best_cluster = g
                        if len(value_counts) > 0:
                            best_match = value_counts.index[0][1]
                        else:
                            best_match = matches_available[0] # Just take a random cluster
                # Use best cluster
                clusters_match.append(best_cluster)
                matches_match.append(best_match)
                del clusters_available[clusters_available.index(best_cluster)]
                del matches_available[matches_available.index(best_match)]
            match_df = pd.DataFrame({'psi': clusters_match, 'psi_est': matches_match}, index=np.arange(nk))
            # replace_df = replace_df.groupby('psi').apply(lambda a: a.value_counts().index[0][1])

            clusters_merged = pd.merge(pd.DataFrame({'psi': clusters_true}), match_df, how='left', on='psi')

            wrong_cluster = np.sum(clusters_merged['psi_est'] != clusters_estimated)
            if measure == 'quantile_all':
                bound = 5000 # 10% error
            elif measure == 'quantile_firm_small':
                bound = 10000 # 20% error
            elif measure == 'quantile_firm_large':
                bound = 10000 # 20% error
            if stayers_movers == 'stayers':
                bound = 35000 # 70% error

            assert wrong_cluster < bound, 'error is {} for {}'.format(wrong_cluster, measure)

def test_cluster_5():
    # Test cluster function works with moments-quantiles options.
    worker_data = []
    worker_data.append({'j': 0, 't': 1, 'i': 0, 'y': 2., 'g': 1})
    worker_data.append({'j': 1, 't': 2, 'i': 0, 'y': 1., 'g': 2})
    worker_data.append({'j': 1, 't': 1, 'i': 1, 'y': 1., 'g': 2})
    worker_data.append({'j': 2, 't': 2, 'i': 1, 'y': 1., 'g': 1})
    worker_data.append({'j': 2, 't': 3, 'i': 1, 'y': 2., 'g': 1})
    worker_data.append({'j': 5, 't': 4, 'i': 1, 'y': 1., 'g': 3})
    worker_data.append({'j': 2, 't': 1, 'i': 3, 'y': 1., 'g': 1})
    worker_data.append({'j': 2, 't': 2, 'i': 3, 'y': 1., 'g': 1})
    worker_data.append({'j': 0, 't': 1, 'i': 4, 'y': 1., 'g': 1})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])[['i', 'j', 'y', 't', 'g']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()

    measures = bpd.measures.moments(measures='mean')
    grouping = bpd.grouping.quantiles(n_quantiles=3)
    bdf = bdf.cluster(measures=measures, grouping=grouping)

    # Clusters:
    # j 0 1 2 3
    # g 2 0 1 0

    assert bdf.iloc[0]['g'] == 2
    assert bdf.iloc[1]['g'] == 0
    assert bdf.iloc[2]['g'] == 0
    assert bdf.iloc[3]['g'] == 1
    assert bdf.iloc[4]['g'] == 1
    assert bdf.iloc[5]['g'] == 0
    assert bdf.iloc[6]['g'] == 1
    assert bdf.iloc[7]['g'] == 1
    assert bdf.iloc[8]['g'] == 2
