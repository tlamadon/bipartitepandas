'''
Tests for bipartitepandas

DATE: March 2021
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
from pytwoway import SimTwoWay

###################################
##### Tests for BipartiteBase #####
###################################

def test_refactor_1():
    # Continuous time, 2 movers between firms 1 and 2, and 1 stayer at firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 0, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 2, 'comp': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']][['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 0
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.shape[0] == 0

def test_refactor_2():
    # Discontinuous time, 2 movers between firms 1 and 2, and 1 stayer at firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 3, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 0, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 2, 'comp': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 0
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

def test_refactor_3():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 2, 'comp': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 2
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 2
    assert movers.iloc[2]['f2i'] == 1
    assert movers.iloc[2]['wid'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 2

def test_refactor_4():
    # Continuous time, 1 mover between firms 1 and 2 and then 2 and 1, 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2, 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1, 'index': 1})
    worker_data.append({'fid': 0, 'year': 3, 'wid': 0, 'comp': 1, 'index': 2})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1, 'index': 3})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1, 'index': 4})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1, 'index': 5})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 2, 'comp': 2, 'index': 6})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 0
    assert movers.iloc[1]['wid'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 1
    assert movers.iloc[2]['f2i'] == 2
    assert movers.iloc[2]['wid'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 2
    assert movers.iloc[3]['f2i'] == 1
    assert movers.iloc[3]['wid'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_5():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1, 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 0, 'year': 4, 'wid': 0, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'index': 5})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 2, 'comp': 2., 'index': 6})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 0
    assert movers.iloc[1]['wid'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 1
    assert movers.iloc[2]['f2i'] == 2
    assert movers.iloc[2]['wid'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 2
    assert movers.iloc[3]['f2i'] == 1
    assert movers.iloc[3]['wid'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_6():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are continuous), 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 0, 'year': 2, 'wid': 0, 'comp': 2., 'index': 1})
    worker_data.append({'fid': 1, 'year': 3, 'wid': 0, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 0, 'year': 5, 'wid': 0, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 5})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'index': 6})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 2, 'comp': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 0
    assert movers.iloc[1]['wid'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 1
    assert movers.iloc[2]['f2i'] == 2
    assert movers.iloc[2]['wid'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 2
    assert movers.iloc[3]['f2i'] == 1
    assert movers.iloc[3]['wid'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_7():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are discontinuous), 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 0, 'year': 3, 'wid': 0, 'comp': 2., 'index': 1})
    worker_data.append({'fid': 1, 'year': 4, 'wid': 0, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 0, 'year': 6, 'wid': 0, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 5})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'index': 6})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 2, 'comp': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 0
    assert movers.iloc[1]['wid'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 1
    assert movers.iloc[2]['f2i'] == 2
    assert movers.iloc[2]['wid'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 2
    assert movers.iloc[3]['f2i'] == 1
    assert movers.iloc[3]['wid'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_8():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are discontinuous), 1 between firms 1 and 2, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 0, 'year': 3, 'wid': 0, 'comp': 2., 'index': 1})
    worker_data.append({'fid': 1, 'year': 4, 'wid': 0, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 0, 'year': 6, 'wid': 0, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 0, 'year': 1, 'wid': 1, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 1, 'comp': 1., 'index': 5})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'index': 6})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 2, 'comp': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 0
    assert movers.iloc[1]['wid'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 0
    assert movers.iloc[2]['f2i'] == 1
    assert movers.iloc[2]['wid'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 2
    assert movers.iloc[3]['f2i'] == 1
    assert movers.iloc[3]['wid'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_9():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are discontinuous), 1 between firms 2 and 1, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 0, 'year': 3, 'wid': 0, 'comp': 2., 'index': 1})
    worker_data.append({'fid': 1, 'year': 4, 'wid': 0, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 0, 'year': 6, 'wid': 0, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 0, 'year': 2, 'wid': 1, 'comp': 1., 'index': 5})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'index': 6})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 2, 'comp': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 0
    assert movers.iloc[1]['wid'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 1
    assert movers.iloc[2]['f2i'] == 0
    assert movers.iloc[2]['wid'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 2
    assert movers.iloc[3]['f2i'] == 1
    assert movers.iloc[3]['wid'] == 2
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_10():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 3, and 1 stayer at firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 2, 'comp': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 2
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.iloc[0]['f1i'] == 2
    assert stayers.iloc[0]['f2i'] == 2
    assert stayers.iloc[0]['wid'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

def test_contiguous_fids_11():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 4, and 1 stayer at firm 4, firm 4 gets reset to firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 3, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 3, 'year': 1, 'wid': 2, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 3, 'year': 2, 'wid': 2, 'comp': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 2
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.iloc[0]['f1i'] == 2
    assert stayers.iloc[0]['f2i'] == 2
    assert stayers.iloc[0]['wid'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1


def test_contiguous_wids_12():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 4, and 1 stayer at firm 4, firm 4 gets reset to firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 3, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 3, 'comp': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 2
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.iloc[0]['f1i'] == 2
    assert stayers.iloc[0]['f2i'] == 2
    assert stayers.iloc[0]['wid'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

def test_contiguous_cids_13():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 4, and 1 stayer at firm 4, firm 4 gets reset to firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'j': 1, 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'j': 2, 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'j': 2, 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'j': 1, 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'j': 1, 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 2, 'comp': 1., 'j': 1, 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year', 'j']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 2
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0

    assert stayers.iloc[0]['f1i'] == 2
    assert stayers.iloc[0]['f2i'] == 2
    assert stayers.iloc[0]['wid'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['j1'] == 0
    assert stayers.iloc[0]['j2'] == 0

def test_contiguous_cids_14():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 4, and 1 stayer at firm 4, firm 4 gets reset to firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'j': 2, 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'j': 1, 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'j': 1, 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'j': 2, 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'j': 2, 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 2, 'comp': 1., 'j': 2, 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year', 'j']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1
    assert movers.iloc[0]['j1'] == 1
    assert movers.iloc[0]['j2'] == 0

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 2
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1
    assert movers.iloc[1]['j1'] == 0
    assert movers.iloc[1]['j2'] == 1

    assert stayers.iloc[0]['f1i'] == 2
    assert stayers.iloc[0]['f2i'] == 2
    assert stayers.iloc[0]['wid'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['j1'] == 1
    assert stayers.iloc[0]['j2'] == 1

def test_col_dict_15():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 4, and 1 stayer at firm 4, firm 4 gets reset to firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 3, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 3, 'comp': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']].rename({'fid': 'firm', 'wid': 'worker'}, axis=1)

    bdf = bpd.BipartiteLong(data=df, col_dict={'fid': 'firm', 'wid': 'worker'})
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 2
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.iloc[0]['f1i'] == 2
    assert stayers.iloc[0]['f2i'] == 2
    assert stayers.iloc[0]['wid'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

def test_worker_year_unique_16():
    # Workers with multiple jobs in the same year, keep the highest paying
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 3, 'year': 2, 'wid': 1, 'comp': 0.5, 'index': 4})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 3, 'comp': 1., 'index': 5})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 3, 'comp': 1.5, 'index': 6})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 3, 'comp': 1., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data().gen_m()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['fid'] == 0
    assert movers.iloc[0]['comp'] == 2
    assert movers.iloc[0]['year'] == 1

    assert movers.iloc[1]['wid'] == 0
    assert movers.iloc[1]['fid'] == 1
    assert movers.iloc[1]['comp'] == 1
    assert movers.iloc[1]['year'] == 2

    assert movers.iloc[2]['wid'] == 1
    assert movers.iloc[2]['fid'] == 1
    assert movers.iloc[2]['comp'] == 1
    assert movers.iloc[2]['year'] == 1

    assert movers.iloc[3]['wid'] == 1
    assert movers.iloc[3]['fid'] == 2
    assert movers.iloc[3]['comp'] == 1
    assert movers.iloc[3]['year'] == 2

    assert movers.iloc[4]['wid'] == 2
    assert movers.iloc[4]['fid'] == 1
    assert movers.iloc[4]['comp'] == 1.5
    assert movers.iloc[4]['year'] == 1

    assert movers.iloc[5]['wid'] == 2
    assert movers.iloc[5]['fid'] == 2
    assert movers.iloc[5]['comp'] == 1
    assert movers.iloc[5]['year'] == 2

def test_general_methods_17():
    # Test some general methods, like n_workers/n_firms/n_clusters, included_cols(), drop(), and rename().
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'j': 2, 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'j': 1, 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'j': 1, 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'j': 2, 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 2, 'comp': 1., 'j': 2, 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 2, 'comp': 1., 'j': 2, 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year', 'j']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    assert bdf.n_workers() == 3
    assert bdf.n_firms() == 3
    assert bdf.n_clusters() == 2

    correct_cols = True
    all_cols = bdf.included_cols()
    for col in ['fid', 'year', 'wid', 'comp', 'j']:
        if col not in all_cols:
            correct_cols = False
            break
    assert correct_cols

    bdf.drop('j1')
    assert 'j1' in bdf.columns and 'j2' in bdf.columns

    bdf.drop('j')
    assert 'j1' not in bdf.columns and 'j2' not in bdf.columns

    bdf.rename({'wid': 'w'})
    assert 'wid' in bdf.columns

    bdf['j1'] = 1
    bdf['j2'] = 1
    bdf.col_dict['j1'] = 'j1'
    bdf.col_dict['j2'] = 'j2'
    assert 'j1' in bdf.columns and 'j2' in bdf.columns
    bdf.rename({'j': 'r'})
    assert 'j1' not in bdf.columns and 'j2' not in bdf.columns

def test_cluster_18():
    '''
    Test cluster function is working correctly.
    '''
    nk = 10
    sim_data = SimTwoWay({'nk': nk}).sim_network()
    bdf = bpd.BipartiteLong(sim_data)
    bdf = bdf.clean_data()
    for grouping in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        print(grouping)
        bdf = bdf.cluster(user_cluster={'grouping': grouping, 'user_KMeans': {'n_clusters': nk}})

        clusters_true = sim_data['psi'].astype('category').cat.codes.astype(int)
        clusters_estimated = bdf['j'].astype(int)

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
            for j in clusters_available: # Iterate over remaining clusters
                cluster_df = replace_df[replace_df['psi'] == j]
                value_counts = cluster_df[cluster_df['psi_est'].isin(matches_available)].value_counts() # Only show valid remaining matches
                if len(value_counts) > 0:
                    proportion = value_counts.iloc[0] / len(cluster_df)
                else:
                    proportion = 0
                if proportion > best_proportion:
                    best_proportion = proportion
                    best_cluster = j
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
        if grouping == 'quantile_all':
            bound = 2500 # 5% error

        elif grouping == 'quantile_firm_small':
            bound = 3000 # 6% error

        elif grouping == 'quantile_firm_large':
            bound = 3000 # 6% error

        assert wrong_cluster < bound, 'error is {} for {}'.format(wrong_cluster, grouping)

############################################
##### Tests for BipartiteLongCollapsed #####
############################################

def test_long_collapsed_1():
    # Test constructor for BipartiteLongCollapsed
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 3, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 3, 'comp': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    df = pd.DataFrame(bdf.get_collapsed_long()).rename({'comp': 'y'}, axis=1)
    bdf = bpd.BipartiteLongCollapsed(df, col_dict={'comp': 'y'})
    bdf = bdf.clean_data()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['fid'] == 0
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['comp'] == 2

    assert movers.iloc[1]['fid'] == 1
    assert movers.iloc[1]['wid'] == 0
    assert movers.iloc[1]['comp'] == 1

    assert movers.iloc[2]['fid'] == 1
    assert movers.iloc[2]['wid'] == 1
    assert movers.iloc[2]['comp'] == 1

    assert movers.iloc[3]['fid'] == 2
    assert movers.iloc[3]['wid'] == 1
    assert movers.iloc[3]['comp'] == 1

    assert stayers.iloc[0]['fid'] == 2
    assert stayers.iloc[0]['wid'] == 2
    assert stayers.iloc[0]['comp'] == 1

#########################################
##### Tests for BipartiteEventStudy #####
#########################################

def test_event_study_1():
    # Test constructor for BipartiteEventStudy
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 3, 'comp': 1., 'index': 4})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 3, 'comp': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    df = pd.DataFrame(bdf.get_es()).rename({'year_1': 'year'}, axis=1)
    bdf = bpd.BipartiteEventStudy(df, col_dict={'year_1': 'year'})
    bdf = bdf.clean_data()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1
    assert movers.iloc[0]['year_1'] == 1
    assert movers.iloc[0]['year_2'] == 2

    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1
    assert movers.iloc[1]['year_1'] == 1
    assert movers.iloc[1]['year_2'] == 2

    assert stayers.iloc[0]['wid'] == 2
    assert stayers.iloc[0]['f1i'] == 2
    assert stayers.iloc[0]['f2i'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['year_1'] == 1
    assert stayers.iloc[0]['year_2'] == 1

    assert stayers.iloc[1]['wid'] == 2
    assert stayers.iloc[1]['f1i'] == 2
    assert stayers.iloc[1]['f2i'] == 2
    assert stayers.iloc[1]['y1'] == 2
    assert stayers.iloc[1]['y2'] == 2
    assert stayers.iloc[1]['year_1'] == 2
    assert stayers.iloc[1]['year_2'] == 2

##################################################
##### Tests for BipartiteEventStudyCollapsed #####
##################################################

def test_event_study_collapsed_1():
    # Test constructor for BipartiteEventStudyCollapsed
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'index': 3})
    worker_data.append({'fid': 2, 'year': 3, 'wid': 1, 'comp': 2., 'index': 4})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 3, 'comp': 1., 'index': 5})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 3, 'comp': 1., 'index': 6})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year']]

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    df = pd.DataFrame(bdf.get_es()).rename({'y1': 'comp1'}, axis=1)
    bdf = bpd.BipartiteEventStudyCollapsed(df, col_dict={'y1': 'comp1'})
    bdf = bdf.clean_data()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['f1i'] == 0
    assert movers.iloc[0]['f2i'] == 1
    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1
    assert movers.iloc[0]['year_start_1'] == 1
    assert movers.iloc[0]['year_end_1'] == 1
    assert movers.iloc[0]['year_start_2'] == 2
    assert movers.iloc[0]['year_end_2'] == 2

    assert movers.iloc[1]['f1i'] == 1
    assert movers.iloc[1]['f2i'] == 2
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1.5
    assert movers.iloc[1]['year_start_1'] == 1
    assert movers.iloc[1]['year_end_1'] == 1
    assert movers.iloc[1]['year_start_2'] == 2
    assert movers.iloc[1]['year_end_2'] == 3

    assert stayers.iloc[0]['f1i'] == 2
    assert stayers.iloc[0]['f2i'] == 2
    assert stayers.iloc[0]['wid'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['year_start_1'] == 1
    assert stayers.iloc[0]['year_end_1'] == 2
    assert stayers.iloc[0]['year_start_2'] == 1
    assert stayers.iloc[0]['year_end_2'] == 2

#####################################
##### Tests for BipartitePandas #####
#####################################

def test_reformatting_1():
    # Convert from long --> event study --> long --> collapsed long --> collapsed event study --> collapsed long to ensure conversion maintains data properly.
    worker_data = []
    worker_data.append({'fid': 0, 'year': 1, 'wid': 0, 'comp': 2., 'j': 1, 'index': 0})
    worker_data.append({'fid': 1, 'year': 2, 'wid': 0, 'comp': 1., 'j': 2, 'index': 1})
    worker_data.append({'fid': 1, 'year': 1, 'wid': 1, 'comp': 1., 'j': 2, 'index': 2})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 1, 'comp': 1., 'j': 1, 'index': 3})
    worker_data.append({'fid': 2, 'year': 3, 'wid': 1, 'comp': 2., 'j': 1, 'index': 4})
    worker_data.append({'fid': 5, 'year': 3, 'wid': 1, 'comp': 1., 'j': 3, 'index': 5})
    worker_data.append({'fid': 2, 'year': 1, 'wid': 3, 'comp': 1., 'j': 1, 'index': 6})
    worker_data.append({'fid': 2, 'year': 2, 'wid': 3, 'comp': 1., 'j': 1, 'index': 7})
    worker_data.append({'fid': 0, 'year': 1, 'wid': 4, 'comp': 1., 'j': 1, 'index': 8})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])[['wid', 'fid', 'comp', 'year', 'j']]

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
    bdf = bdf.get_collapsed_long()
    bdf = bdf.clean_data()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] == 1]

    assert movers.iloc[0]['wid'] == 0
    assert movers.iloc[0]['fid'] == 0
    assert movers.iloc[0]['comp'] == 2
    assert movers.iloc[0]['year_start'] == 1
    assert movers.iloc[0]['year_end'] == 1
    assert movers.iloc[0]['j'] == 0

    assert movers.iloc[1]['wid'] == 0
    assert movers.iloc[1]['fid'] == 1
    assert movers.iloc[1]['comp'] == 1
    assert movers.iloc[1]['year_start'] == 2
    assert movers.iloc[1]['year_end'] == 2
    assert movers.iloc[1]['j'] == 1

    assert movers.iloc[2]['wid'] == 1
    assert movers.iloc[2]['fid'] == 1
    assert movers.iloc[2]['comp'] == 1
    assert movers.iloc[2]['year_start'] == 1
    assert movers.iloc[2]['year_end'] == 1
    assert movers.iloc[2]['j'] == 1

    assert movers.iloc[3]['wid'] == 1
    assert movers.iloc[3]['fid'] == 2
    assert movers.iloc[3]['comp'] == 1.5
    assert movers.iloc[3]['year_start'] == 2
    assert movers.iloc[3]['year_end'] == 3
    assert movers.iloc[3]['j'] == 0

    assert stayers.iloc[0]['wid'] == 2
    assert stayers.iloc[0]['fid'] == 2
    assert stayers.iloc[0]['comp'] == 1
    assert stayers.iloc[0]['year_start'] == 1
    assert stayers.iloc[0]['year_end'] == 2
    assert stayers.iloc[0]['j'] == 0

    assert stayers.iloc[1]['wid'] == 3
    assert stayers.iloc[1]['fid'] == 0
    assert stayers.iloc[1]['comp'] == 1
    assert stayers.iloc[1]['year_start'] == 1
    assert stayers.iloc[1]['year_end'] == 1
    assert stayers.iloc[1]['j'] == 0
