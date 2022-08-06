'''
Tests for bipartitepandas.
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pickle

###################################
##### Tests for BipartiteBase #####
###################################

def test_refactor_1():
    # 2 movers between firms 0 and 1, and 1 stayer at firm 2.
    worker_data = []
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Firm 1 -> 0
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 0, 'y': 1., 't': 2})
    # Firm 2 -> 2
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 2, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean(bpd.clean_params({'connectedness': 'connected'}))
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

def test_refactor_2():
    # 2 movers between firms 0 and 1, and 1 stayer at firm 2. Time has jumps.
    worker_data = []
    # Firm 0 -> 1
    # Time 1 -> 3
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 3})
    # Firm 1 -> 0
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 0, 'y': 1., 't': 2})
    # Firm 2 -> 2
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 2, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean(bpd.clean_params({'connectedness': 'connected'}))
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

def test_refactor_3():
    # 1 mover between firms 0 and 1, and 2 between firms 1 and 2.
    worker_data = []
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['i'] == 2
    assert movers.iloc[2]['j1'] == 2
    assert movers.iloc[2]['j2'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 2

def test_refactor_4():
    # 1 mover between firms 0 and 1, and 2 between firms 1 and 2.
    worker_data = []
    # Firm 0 -> 1 -> 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 3})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j1'] == 1
    assert movers.iloc[2]['j2'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_5():
    # 1 mover between firms 0 and 1, and 2 between firms 1 and 2. Time has jumps.
    worker_data = []
    # Firm 0 -> 1 -> 0
    # Time 1 -> 2 -> 4
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j1'] == 1
    assert movers.iloc[2]['j2'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_6():
    # 1 mover between firms 0 and 1, and 2 between firms 1 and 2. Time has jumps.
    worker_data = []
    # Firm 0 -> 0 -> 1 -> 0
    # Time 1 -> 2 -> 3 -> 5
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 2})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 3})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 5})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j1'] == 1
    assert movers.iloc[2]['j2'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_7():
    # 1 mover between firms 0 and 1, and 2 between firms 1 and 2. Time has jumps.
    worker_data = []
    # Firm 0 -> 0 -> 1 -> 0
    # Time 1 -> 3 -> 4 -> 6
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 3})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 4})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 6})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j1'] == 1
    assert movers.iloc[2]['j2'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_8():
    # 2 movers between firms 0 and 1, and 1 between firms 1 and 2. Time has jumps.
    worker_data = []
    # Firm 0 -> 0 -> 1 -> 0
    # Time 1 -> 3 -> 4 -> 6
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 3})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 4})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 6})
    # Firm 0 -> 1
    worker_data.append({'i': 1, 'j': 0, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j1'] == 0
    assert movers.iloc[2]['j2'] == 1
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_9():
    # 2 movers between firms 0 and 1, and 1 between firms 1 and 2. Time has jumps.
    worker_data = []
    # Firm 0 -> 0 -> 1 -> 0
    # Time 1 -> 3 -> 4 -> 6
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 3})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 4})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 6})
    # Firm 1 -> 0
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 0, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 0
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j1'] == 1
    assert movers.iloc[2]['j2'] == 0
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['j1'] == 2
    assert movers.iloc[3]['j2'] == 1
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_refactor_10():
    # 1 mover between firms 0 and 1, 1 between firms 1 and 2, and 1 stayer at firm 2.
    worker_data = []
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 2
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

def test_refactor_11():
    # 1 mover between firms 0 and 1 and 2 and 3, 1 between firms 1 and 2, and 1 stayer at firm 2.
    # Check going to event study and back to long, for data where movers have extended periods where they stay at the same firm
    worker_data = []
    # Firm 0 -> 1 -> 2 -> 3
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    worker_data.append({'i': 0, 'j': 2, 'y': 0.5, 't': 3})
    worker_data.append({'i': 0, 'j': 2, 'y': 0.5, 't': 4})
    worker_data.append({'i': 0, 'j': 2, 'y': 0.75, 't': 5})
    worker_data.append({'i': 0, 'j': 3, 'y': 1.5, 't': 6})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 2
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df).clean().to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 0
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 0.5
    assert stayers.iloc[0]['y2'] == 0.5
    assert stayers.iloc[0]['t1'] == 3
    assert stayers.iloc[0]['t2'] == 4

    assert stayers.iloc[1]['i'] == 0
    assert stayers.iloc[1]['j1'] == 2
    assert stayers.iloc[1]['j2'] == 2
    assert stayers.iloc[1]['y1'] == 0.5
    assert stayers.iloc[1]['y2'] == 0.75
    assert stayers.iloc[1]['t1'] == 4
    assert stayers.iloc[1]['t2'] == 5

    assert stayers.iloc[2]['i'] == 2
    assert stayers.iloc[2]['j1'] == 2
    assert stayers.iloc[2]['j2'] == 2
    assert stayers.iloc[2]['y1'] == 1.
    assert stayers.iloc[2]['y2'] == 1.
    assert stayers.iloc[2]['t1'] == 1
    assert stayers.iloc[2]['t2'] == 1

    assert stayers.iloc[3]['i'] == 2
    assert stayers.iloc[3]['j1'] == 2
    assert stayers.iloc[3]['j2'] == 2
    assert stayers.iloc[3]['y1'] == 1.
    assert stayers.iloc[3]['y2'] == 1.
    assert stayers.iloc[3]['t1'] == 2
    assert stayers.iloc[3]['t2'] == 2

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2.
    assert movers.iloc[0]['y2'] == 1.
    assert movers.iloc[0]['t1'] == 1
    assert movers.iloc[0]['t2'] == 2

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['y1'] == 1.
    assert movers.iloc[1]['y2'] == 0.5
    assert movers.iloc[1]['t1'] == 2
    assert movers.iloc[1]['t2'] == 3

    assert movers.iloc[2]['i'] == 0
    assert movers.iloc[2]['j1'] == 2
    assert movers.iloc[2]['j2'] == 3
    assert movers.iloc[2]['y1'] == 0.75
    assert movers.iloc[2]['y2'] == 1.5
    assert movers.iloc[2]['t1'] == 5
    assert movers.iloc[2]['t2'] == 6

    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['j1'] == 1
    assert movers.iloc[3]['j2'] == 2
    assert movers.iloc[3]['y1'] == 1.
    assert movers.iloc[3]['y2'] == 1.
    assert movers.iloc[3]['t1'] == 1
    assert movers.iloc[3]['t2'] == 2

    bdf = bdf.to_long()

    assert np.all(bdf[['i', 'j', 'y', 't']].to_numpy() == df.to_numpy())

def test_refactor_12():
    # Check going to event study and back to long
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()

    assert len(bdf) == len(bdf.to_eventstudy().to_long())

def test_refactor_13():
    # Check collapsing, uncollapsing, then recollapsing
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(3456))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean().collapse()

    assert len(bdf) == len(bdf.uncollapse().collapse())

def test_contiguous_fids_11():
    # Check _make_categorical_contiguous() with firm ids.
    worker_data = []
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Firm 1 -> 3
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 3, 'y': 1., 't': 2})
    # Firm 3 -> 3
    worker_data.append({'i': 2, 'j': 3, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 3, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

def test_contiguous_wids_12():
    # Check _make_categorical_contiguous() with worker ids.
    worker_data = []
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Worker 3.5
    # Firm 2 -> 2
    worker_data.append({'i': 3.5, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3.5, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

def test_contiguous_cids_13():
    # Check _make_categorical_contiguous() with cluster ids.
    worker_data = []
    # Firm 0 -> 1
    # Cluster 1 -> 2
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'g': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2, 'g': 2})
    # Firm 1 -> 2
    # Cluster 2 -> 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'g': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'g': 1})
    # Firm 2 -> 2
    # Cluster 1 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1, 'g': 1})
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 2, 'g': 1})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['g1'] == 0
    assert stayers.iloc[0]['g2'] == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1
    assert movers.iloc[0]['g1'] == 0
    assert movers.iloc[0]['g2'] == 1

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1
    assert movers.iloc[1]['g1'] == 1
    assert movers.iloc[1]['g2'] == 0

def test_contiguous_cids_14():
    # Check _make_categorical_contiguous() with cluster ids.
    worker_data = []
    # Firm 0 -> 1
    # Cluster 2 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'g': 2})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2, 'g': 1})
    # Firm 1 -> 2
    # Cluster 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'g': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'g': 2})
    # Firm 2 -> 2
    # Cluster 2 -> 2
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1, 'g': 2})
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 2, 'g': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df, track_id_changes=True)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy().original_ids()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j1'] == 0
    assert movers.iloc[0]['j2'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1
    assert movers.iloc[0]['original_g1'] == 2
    assert movers.iloc[0]['original_g2'] == 1
    assert movers.iloc[0]['g1'] == 0
    assert movers.iloc[0]['g2'] == 1

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j1'] == 1
    assert movers.iloc[1]['j2'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1
    assert movers.iloc[1]['original_g1'] == 1
    assert movers.iloc[1]['original_g2'] == 2
    assert movers.iloc[1]['g1'] == 1
    assert movers.iloc[1]['g2'] == 0

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['original_g1'] == 2
    assert stayers.iloc[0]['original_g2'] == 2
    assert stayers.iloc[0]['g1'] == 0
    assert stayers.iloc[0]['g2'] == 0

# def test_col_dict_15():
#     # Check that col_dict works properly.
#     worker_data = []
#     # Firm 0 -> 1
#     worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
#     worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
#     # Firm 1 -> 2
#     worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
#     worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
#     # Worker 3
#     # Firm 2 -> 2
#     worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
#     worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2})

#     df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)]).rename({'j': 'firm', 'i': 'worker'}, axis=1)

#     bdf = bpd.BipartiteLong(data=df, col_dict={'j': 'firm', 'i': 'worker'})
#     bdf = bdf.clean()
#     bdf = bdf.collapse()
#     bdf = bdf.to_eventstudy()

#     stayers = bdf[bdf['m'] == 0]
#     movers = bdf[bdf['m'] > 0]

#     assert stayers.iloc[0]['i'] == 2
#     assert stayers.iloc[0]['j1'] == 2
#     assert stayers.iloc[0]['j2'] == 2
#     assert stayers.iloc[0]['y1'] == 1
#     assert stayers.iloc[0]['y2'] == 1

#     assert movers.iloc[0]['i'] == 0
#     assert movers.iloc[0]['j1'] == 0
#     assert movers.iloc[0]['j2'] == 1
#     assert movers.iloc[0]['y1'] == 2
#     assert movers.iloc[0]['y2'] == 1

#     assert movers.iloc[1]['i'] == 1
#     assert movers.iloc[1]['j1'] == 1
#     assert movers.iloc[1]['j2'] == 2
#     assert movers.iloc[1]['y1'] == 1
#     assert movers.iloc[1]['y2'] == 1

def test_worker_year_unique_16_1():
    # Workers with multiple jobs in the same year, keep the highest paying, with long format. Testing 'max', 'sum', and 'mean' options, where options should not have an effect.
    worker_data = []
    # Firm 0 -> 1
    # Time 1 -> 2
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Firm 1 -> 2 -> 3
    # Time 1 -> 2 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    worker_data.append({'i': 1, 'j': 3, 'y': 0.5, 't': 2})
    # Worker 3
    # Firm 2 -> 1 -> 2
    # Time 1 -> 1 -> 2
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 1, 'y': 1.5, 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    for how in ['max', 'sum', 'mean']:
        bdf = bpd.BipartiteLong(data=df)
        bdf = bdf.clean(bpd.clean_params({'i_t_how': how}))

        stayers = bdf[bdf['m'] == 0]
        movers = bdf[bdf['m'] > 0]

        assert len(stayers) == 0

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
    # Firm 0 -> 1
    # Time 1 -> 2
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Firm 1 -> 2 -> 2 -> 3
    # Time 1 -> 2 -> 2 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 1.5, 't': 2})
    worker_data.append({'i': 1, 'j': 3, 'y': 0.5, 't': 2})
    # Worker 3
    # Firm 2 -> 1 -> 2
    # Time 1 -> 1 -> 2
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 1, 'y': 1.5, 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    for how in ['max', 'sum', 'mean']:
        bdf = bpd.BipartiteLong(data=df.copy())
        bdf = bdf.clean(bpd.clean_params({'i_t_how': how}))

        stayers = bdf[bdf['m'] == 0]
        movers = bdf[bdf['m'] > 0]

        assert len(stayers) == 0

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
    # Workers with multiple jobs in the same year, keep the highest paying, with collapsed long format. Testing 'max', 'sum', and 'mean' options, where options should have an effect. Using collapsed long data.
    worker_data = []
    # Firm 0 -> 1
    # Time 1 -> 2
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't1': 1, 't2': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't1': 2, 't2': 2})
    # Firm 1 -> 2 -> 2 -> 3
    # Time 1 -> 2 -> 2 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't1': 1, 't2': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't1': 2, 't2': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 1.5, 't1': 2, 't2': 2})
    worker_data.append({'i': 1, 'j': 3, 'y': 0.5, 't1': 2, 't2': 2})
    # Worker 3
    # Firm 2 -> 1
    # Time 1 -> 1
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't1': 1, 't2': 2})
    worker_data.append({'i': 3, 'j': 1, 'y': 1.5, 't1': 1, 't2': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    for how in ['max', 'sum', 'mean']:
        bdf = bpd.BipartiteLongCollapsed(data=df)
        bdf = bdf.clean(bpd.clean_params({'i_t_how': how}))

        stayers = bdf[bdf['m'] == 0]
        movers = bdf[bdf['m'] > 0]

        assert stayers.iloc[0]['i'] == 2
        assert stayers.iloc[0]['j'] == 1
        assert stayers.iloc[0]['y'] == 1.5
        assert stayers.iloc[0]['t1'] == 1
        assert stayers.iloc[0]['t2'] == 2

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

def test_worker_year_unique_16_4():
    # Workers with multiple jobs in the same year, keep the highest paying, with event study format. Testing 'max', 'sum', and 'mean' options, where options should have an effect. NOTE: because of how data converts from event study to long (it only shifts period 2 (e.g. j2, y2) for the last row, as it assumes observations zigzag), it will only correct duplicates for period 1
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j1': 0, 'j2': 1, 'y1': 2., 'y2': 1., 't1': 1, 't2': 2})
    # Worker 1
    worker_data.append({'i': 1, 'j1': 1, 'j2': 2, 'y1': 0.5, 'y2': 1.5, 't1': 1, 't2': 2})
    worker_data.append({'i': 1, 'j1': 1, 'j2': 2, 'y1': 0.75, 'y2': 1., 't1': 1, 't2': 2})
    worker_data.append({'i': 1, 'j1': 2, 'j2': 1, 'y1': 1., 'y2': 2., 't1': 1, 't2': 2})
    # Worker 3
    worker_data.append({'i': 3, 'j1': 2, 'j2': 2, 't1': 1, 't2': 1, 'y1': 1., 'y2': 1.})
    worker_data.append({'i': 3, 'j1': 2, 'j2': 2, 'y1': 1., 'y2': 1., 't1': 2, 't2': 2})
    worker_data.append({'i': 3, 'j1': 1, 'j2': 1, 'y1': 1.5, 'y2': 1.5, 't1': 1, 't2': 1})
    worker_data.append({'i': 3, 'j1': 1, 'j2': 1, 'y1': 1.5, 'y2': 1.5, 't1': 2, 't2': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    for how in ['max', 'sum', 'mean']:
        bdf = bpd.BipartiteEventStudy(data=df.copy(), track_id_changes=True)
        bdf = bdf.clean(bpd.clean_params({'i_t_how': how})).original_ids()

        stayers = bdf[bdf['m'] == 0]
        movers = bdf[bdf['m'] > 0]

        assert stayers.iloc[0]['i'] == 2
        assert stayers.iloc[0]['original_i'] == 3
        assert stayers.iloc[0]['j1'] == 1
        assert stayers.iloc[0]['j2'] == 1
        assert stayers.iloc[0]['y1'] == 1.5
        assert stayers.iloc[0]['y2'] == 1.5
        assert stayers.iloc[0]['t1'] == 1
        assert stayers.iloc[0]['t2'] == 1

        assert stayers.iloc[1]['i'] == 2
        assert stayers.iloc[1]['original_i'] == 3
        assert stayers.iloc[1]['j1'] == 1
        assert stayers.iloc[1]['j2'] == 1
        assert stayers.iloc[1]['y1'] == 1.5
        assert stayers.iloc[1]['y2'] == 1.5
        assert stayers.iloc[1]['t1'] == 2
        assert stayers.iloc[1]['t2'] == 2

        assert movers.iloc[0]['original_i'] == 0
        assert movers.iloc[0]['i'] == 0
        assert movers.iloc[0]['j1'] == 0
        assert movers.iloc[0]['j2'] == 1
        assert movers.iloc[0]['y1'] == 2
        assert movers.iloc[0]['y2'] == 1
        assert movers.iloc[0]['t1'] == 1
        assert movers.iloc[0]['t2'] == 2

        assert movers.iloc[1]['original_i'] == 1
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

def test_string_ids_17():
    # String worker and firm ids.
    worker_data = []
    # Worker 'a'
    worker_data.append({'i': 'a', 'j': 'a', 'y': 2., 't': 1})
    worker_data.append({'i': 'a', 'j': 'b', 'y': 1., 't': 2})
    # Worker 'b'
    worker_data.append({'i': 'b', 'j': 'b', 'y': 1., 't': 1})
    worker_data.append({'i': 'b', 'j': 'c', 'y': 1., 't': 2})
    worker_data.append({'i': 'b', 'j': 'd', 'y': 0.5, 't': 2})
    # Worker 'd'
    worker_data.append({'i': 'd', 'j': 'c', 'y': 1., 't': 1})
    worker_data.append({'i': 'd', 'j': 'b', 'y': 1.5, 't': 1})
    worker_data.append({'i': 'd', 'j': 'c', 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

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
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'g': 2})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2, 'g': 1})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'g': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'g': 2})
    # Worker 2
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1, 'g': 2})
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 2, 'g': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    bdf = bdf.to_eventstudy()

    assert bdf.n_workers() == 3
    assert bdf.n_firms() == 3
    assert bdf.n_clusters() == 2

    correct_cols = True
    all_cols = bdf._included_cols()
    for col in ['i', 'j', 'y', 't', 'g']:
        if col not in all_cols:
            correct_cols = False
            break
    assert correct_cols

    bdf.drop('g1', axis=1, inplace=True, allow_optional=True)
    assert 'g1' in bdf.columns and 'g2' in bdf.columns

    bdf.drop('g', axis=1, inplace=True, allow_optional=True)
    assert 'g1' not in bdf.columns and 'g2' not in bdf.columns

    try:
        bdf.rename({'i': 'w'}, axis=1, inplace=True)
        success = False
    except ValueError:
        success = True

    assert success

    bdf['g1'] = 1
    bdf['g2'] = 1
    # bdf.col_dict['g1'] = 'g1'
    # bdf.col_dict['g2'] = 'g2'
    assert 'g1' in bdf.columns and 'g2' in bdf.columns
    bdf.rename({'g': 'r'}, axis=1, inplace=True, allow_optional=True)
    assert 'g1' not in bdf.columns and 'g2' not in bdf.columns
    assert 'r1' in bdf.columns and 'r2' in bdf.columns

def test_copy_19():
    # Make sure changing attributes in a copied version does not overwrite values in the original.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    worker_data.append({'i': 1, 'j': 3, 'y': 0.5, 't': 2})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 1, 'y': 1.5, 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    # Long
    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean().drop('m', axis=1, inplace=True, allow_optional=True)
    bdf2 = bdf.copy()
    bdf2 = bdf2.gen_m(copy=False)

    assert 'm' in bdf2._included_cols() and 'm' not in bdf._included_cols()

    # Event study
    bdf = bdf.gen_m(copy=False).to_eventstudy()
    bdf = bdf.clean().drop('m', axis=1, inplace=True, allow_optional=True)
    bdf2 = bdf.copy()
    bdf2 = bdf2.gen_m(copy=False)

    assert 'm' in bdf2._included_cols() and 'm' not in bdf._included_cols()

    # Collapsed long
    bdf = bdf.gen_m(copy=False).to_long().collapse()
    bdf = bdf.clean().drop('m', axis=1, inplace=True, allow_optional=True)
    bdf2 = bdf.copy()
    bdf2 = bdf2.gen_m(copy=False)

    assert 'm' in bdf2._included_cols() and 'm' not in bdf._included_cols()

    # Collapsed event study
    bdf = bdf.gen_m(copy=False).to_eventstudy()
    bdf = bdf.clean().drop('m', axis=1, inplace=True, allow_optional=True)
    bdf2 = bdf.copy()
    bdf2 = bdf2.gen_m(copy=False)

    assert 'm' in bdf2._included_cols() and 'm' not in bdf._included_cols()

def test_id_reference_dict_20():
    # String worker and firm ids, link with id_reference_dict.
    worker_data = []
    # Worker 'a'
    worker_data.append({'i': 'a', 'j': 'a', 'y': 2., 't': 1})
    worker_data.append({'i': 'a', 'j': 'b', 'y': 1., 't': 2})
    # Worker 'b'
    worker_data.append({'i': 'b', 'j': 'b', 'y': 1., 't': 1})
    worker_data.append({'i': 'b', 'j': 'c', 'y': 1., 't': 2})
    worker_data.append({'i': 'b', 'j': 'd', 'y': 0.5, 't': 2})
    # Worker 'd'
    worker_data.append({'i': 'd', 'j': 'c', 'y': 1., 't': 1})
    worker_data.append({'i': 'd', 'j': 'b', 'y': 1.5, 't': 1})
    worker_data.append({'i': 'd', 'j': 'c', 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df, track_id_changes=True)
    bdf = bdf.clean()

    id_reference_dict = bdf.id_reference_dict

    merge_df = bdf.merge(id_reference_dict['i'], how='left', left_on='i', right_on='adjusted_ids_1').rename({'original_ids': 'original_i'}, axis=1)
    merge_df = merge_df.merge(id_reference_dict['j'], how='left', left_on='j', right_on='adjusted_ids_1').rename({'original_ids': 'original_j'}, axis=1)

    stayers = merge_df[merge_df['m'] == 0]
    movers = merge_df[merge_df['m'] == 1]

    assert len(stayers) == 0

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
    # Worker 'a'
    worker_data.append({'i': 'a', 'j': 'a', 'y': 2., 't': 1})
    worker_data.append({'i': 'a', 'j': 'b', 'y': 1., 't': 2})
    # Worker 'b'
    worker_data.append({'i': 'b', 'j': 'b', 'y': 1., 't': 1})
    worker_data.append({'i': 'b', 'j': 'c', 'y': 1., 't': 2})
    worker_data.append({'i': 'b', 'j': 'd', 'y': 0.5, 't': 2})
    # Worker 'd'
    worker_data.append({'i': 'd', 'j': 'c', 'y': 1., 't': 1})
    worker_data.append({'i': 'd', 'j': 'b', 'y': 1.5, 't': 1})
    worker_data.append({'i': 'd', 'j': 'c', 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df, track_id_changes=True)
    bdf = bdf.clean()

    merge_df = bdf.original_ids()

    stayers = merge_df[merge_df['m'] == 0]
    movers = merge_df[merge_df['m'] == 1]

    assert len(stayers) == 0

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
    # Worker 'a'
    # Firm a -> b -> c turns into 0 -> 1 -> 2 turns into 0 -> 1
    worker_data.append({'i': 'a', 'j': 'a', 'y': 2., 't': 1})
    worker_data.append({'i': 'a', 'j': 'b', 'y': 1., 't': 2})
    worker_data.append({'i': 'a', 'j': 'c', 'y': 1.5, 't': 3})
    # Worker 'b'
    # Firm b -> d turns into 1 -> 3 turns into 0 -> 2
    worker_data.append({'i': 'b', 'j': 'b', 'y': 1., 't': 1})
    worker_data.append({'i': 'b', 'j': 'd', 'y': 1., 't': 2})
    worker_data.append({'i': 'b', 'j': 'c', 'y': 0.5, 't': 2})
    # Worker 'd'
    # Firm b -> d turns into 1 -> 3 turns into 0 -> 2
    worker_data.append({'i': 'd', 'j': 'd', 'y': 1., 't': 1})
    worker_data.append({'i': 'd', 'j': 'b', 'y': 1.5, 't': 1})
    worker_data.append({'i': 'd', 'j': 'd', 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df, track_id_changes=True)
    bdf = bdf.clean()
    bdf = bdf[bdf['j'] > 0]
    bdf = bdf.clean(bpd.clean_params({'connectedness': None}))

    merge_df = bdf.original_ids()

    stayers = merge_df[merge_df['m'] == 0]
    movers = merge_df[merge_df['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['original_i'] == 'a'
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['original_j'] == 'b'
    assert movers.iloc[0]['y'] == 1
    assert movers.iloc[0]['t'] == 2

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['original_i'] == 'a'
    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['original_j'] == 'c'
    assert movers.iloc[1]['y'] == 1.5
    assert movers.iloc[1]['t'] == 3

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['original_i'] == 'b'
    assert movers.iloc[2]['j'] == 0
    assert movers.iloc[2]['original_j'] == 'b'
    assert movers.iloc[2]['y'] == 1
    assert movers.iloc[2]['t'] == 1

    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['original_i'] == 'b'
    assert movers.iloc[3]['j'] == 2
    assert movers.iloc[3]['original_j'] == 'd'
    assert movers.iloc[3]['y'] == 1
    assert movers.iloc[3]['t'] == 2

    assert movers.iloc[4]['i'] == 2
    assert movers.iloc[4]['original_i'] == 'd'
    assert movers.iloc[4]['j'] == 0
    assert movers.iloc[4]['original_j'] == 'b'
    assert movers.iloc[4]['y'] == 1.5
    assert movers.iloc[4]['t'] == 1

    assert movers.iloc[5]['i'] == 2
    assert movers.iloc[5]['original_i'] == 'd'
    assert movers.iloc[5]['j'] == 2
    assert movers.iloc[5]['original_j'] == 'd'
    assert movers.iloc[5]['y'] == 1
    assert movers.iloc[5]['t'] == 2

def test_fill_time_24_1():
    # Test .fill_time() method for long format, with no data to fill in.
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
    new_df = bdf.fill_missing_periods()

    stayers = new_df[new_df['m'] == 0]
    movers = new_df[new_df['m'] == 1]

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

def test_fill_time_24_2():
    # Test .fill_time() method for long format, with 1 row of data to fill in.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Worker 1
    # Time 1 -> 3
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 3})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    new_df = bdf.fill_missing_periods()

    stayers = new_df[new_df.groupby('i')['m'].transform('max') == 0]
    movers = new_df[new_df.groupby('i')['m'].transform('max') == 1]

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
    assert movers.iloc[3]['j'] == - 1
    assert pd.isna(movers.iloc[3]['y'])
    assert pd.isna(movers.iloc[3]['m'])

    assert movers.iloc[4]['i'] == 1
    assert movers.iloc[4]['j'] == 2
    assert movers.iloc[4]['y'] == 1

def test_fill_time_24_3():
    # Test .fill_time() method for long format, with 2 rows of data to fill in.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Worker 1
    # Time 1 -> 4
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 4})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    new_df = bdf.fill_missing_periods()

    stayers = new_df[new_df.groupby('i')['m'].transform('max') == 0]
    movers = new_df[new_df.groupby('i')['m'].transform('max') == 1]

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
    assert movers.iloc[3]['j'] == - 1
    assert pd.isna(movers.iloc[3]['y'])
    assert pd.isna(movers.iloc[3]['m'])

    assert movers.iloc[4]['i'] == 1
    assert movers.iloc[4]['j'] == - 1
    assert pd.isna(movers.iloc[4]['y'])
    assert pd.isna(movers.iloc[4]['m'])

    assert movers.iloc[5]['i'] == 1
    assert movers.iloc[5]['j'] == 2
    assert movers.iloc[5]['y'] == 1

def test_fill_time_24_4():
    # Test .fill_time() method for long format, with 2 rows of data to fill in, where fill_dict is customized and there is a custom column.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'rs': 3})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2, 'rs': 4})
    # Worker 1
    # Time 1 -> 4
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'rs': 1.5})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 4, 'rs': 2})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1, 'rs': 0.25})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2, 'rs': -5})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    print('df.columns:', df.columns)

    bdf = bpd.BipartiteDataFrame(df)
    bdf = bdf.clean()
    new_df = bdf.fill_missing_periods({'y': 'hello!', 'rs': 'how do you do?'})

    stayers = new_df[new_df.groupby('i')['m'].transform('max') == 0]
    movers = new_df[new_df.groupby('i')['m'].transform('max') == 1]

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j'] == 2
    assert stayers.iloc[0]['y'] == 1
    assert stayers.iloc[0]['rs'] == 0.25

    assert stayers.iloc[1]['i'] == 2
    assert stayers.iloc[1]['j'] == 2
    assert stayers.iloc[1]['y'] == 1
    assert stayers.iloc[1]['rs'] == -5

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['y'] == 2
    assert movers.iloc[0]['rs'] == 3

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['y'] == 1
    assert movers.iloc[1]['rs'] == 4

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['y'] == 1
    assert movers.iloc[2]['rs'] == 1.5

    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['j'] == - 1
    assert movers.iloc[3]['y'] == 'hello!'
    assert pd.isna(movers.iloc[3]['m'])
    assert movers.iloc[3]['rs'] == 'how do you do?'

    assert movers.iloc[4]['i'] == 1
    assert movers.iloc[4]['j'] == - 1
    assert movers.iloc[4]['y'] == 'hello!'
    assert pd.isna(movers.iloc[4]['m'])
    assert movers.iloc[4]['rs'] == 'how do you do?'

    assert movers.iloc[5]['i'] == 1
    assert movers.iloc[5]['j'] == 2
    assert movers.iloc[5]['y'] == 1
    assert movers.iloc[5]['rs'] == 2

def test_uncollapse_25():
    # Convert from collapsed long to long format.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't1': 1, 't2': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't1': 2, 't2': 2})
    # Worker 1
    # Time 1 -> 3
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't1': 1, 't2': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't1': 2, 't2': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 1.5, 't1': 2, 't2': 2})
    worker_data.append({'i': 1, 'j': 3, 'y': 0.5, 't1': 2, 't2': 2})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't1': 1, 't2': 2})
    worker_data.append({'i': 3, 'j': 1, 'y': 1.5, 't1': 1, 't2': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

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

def test_keep_ids_26():
    # Keep only given ids.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()
    all_fids = bdf['j'].unique()
    ids_to_keep = all_fids[: len(all_fids) // 2]
    bdf_keep = bdf.to_eventstudy().keep_ids('j', ids_to_keep).to_long()
    assert set(bdf_keep['j']) == set(ids_to_keep)

    # Make sure long and es give same results
    bdf_keep2 = bdf.keep_ids('j', ids_to_keep)
    assert len(bdf_keep) == len(bdf_keep2)

def test_drop_ids_27():
    # Drop given ids.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()
    all_fids = bdf['j'].unique()
    ids_to_drop = all_fids[: len(all_fids) // 2]
    bdf_keep = bdf.to_eventstudy().drop_ids('j', ids_to_drop).to_long()
    assert set(bdf_keep['j']) == set(all_fids).difference(set(ids_to_drop))

    # Make sure long and es give same results
    bdf_keep2 = bdf.drop_ids('j', ids_to_drop)
    assert len(bdf_keep) == len(bdf_keep2)

def test_drop_returns_28_1():
    # Drop observations where a worker leaves a firm then returns to it
    worker_data = []
    # Firm 0 -> 1 -> 0
    # Time 1 -> 2 -> 4
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean(bpd.clean_params({'drop_returns': 'returns'}))

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 0
    assert stayers.iloc[0]['j'] == 0
    assert stayers.iloc[0]['y'] == 1
    assert stayers.iloc[0]['t'] == 2

    assert movers.iloc[0]['i'] == 1
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['y'] == 1
    assert movers.iloc[0]['t'] == 1

    assert movers.iloc[1]['i'] == 1
    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['y'] == 1
    assert movers.iloc[1]['t'] == 2

    assert movers.iloc[2]['i'] == 2
    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['y'] == 1
    assert movers.iloc[2]['t'] == 1

    assert movers.iloc[3]['i'] == 2
    assert movers.iloc[3]['j'] == 0
    assert movers.iloc[3]['y'] == 2
    assert movers.iloc[3]['t'] == 2

def test_drop_returns_28_2():
    # Drop workers who ever leave a firm then return to it
    worker_data = []
    # Firm 0 -> 1 -> 0
    # Time 1 -> 2 -> 4
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean(bpd.clean_params({'drop_returns': 'returners'}))

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert len(stayers) == 0

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['y'] == 1
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
    assert movers.iloc[3]['j'] == 0
    assert movers.iloc[3]['y'] == 2
    assert movers.iloc[3]['t'] == 2

def test_drop_returns_28_3():
    # Keep first spell where a worker leaves a firm then returns to it
    worker_data = []
    # Firm 0 -> 1 -> 0
    # Time 1 -> 2 -> 3 -> 5 -> 6
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 0, 'y': 3., 't': 2})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 3})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 5})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 6})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean(bpd.clean_params({'drop_returns': 'keep_first_returns'}))

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 0
    assert stayers.iloc[0]['j'] == 0
    assert stayers.iloc[0]['y'] == 2
    assert stayers.iloc[0]['t'] == 1

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j'] == 0
    assert movers.iloc[0]['y'] == 3
    assert movers.iloc[0]['t'] == 2

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j'] == 1
    assert movers.iloc[1]['y'] == 1
    assert movers.iloc[1]['t'] == 3

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['y'] == 1
    assert movers.iloc[2]['t'] == 1

    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['j'] == 2
    assert movers.iloc[3]['y'] == 1
    assert movers.iloc[3]['t'] == 2

    assert movers.iloc[4]['i'] == 2
    assert movers.iloc[4]['j'] == 2
    assert movers.iloc[4]['y'] == 1
    assert movers.iloc[4]['t'] == 1

    assert movers.iloc[5]['i'] == 2
    assert movers.iloc[5]['j'] == 1
    assert movers.iloc[5]['y'] == 2
    assert movers.iloc[5]['t'] == 2

def test_drop_returns_28_4():
    # Keep last spell where a worker leaves a firm then returns to it
    worker_data = []
    # Firm 0 -> 1 -> 0
    # Time 1 -> 2 -> 3 -> 5 -> 6
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 0, 'y': 3., 't': 2})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 3})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 5})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 6})
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Firm 2 -> 1
    worker_data.append({'i': 2, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 2, 'j': 1, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean(bpd.clean_params({'drop_returns': 'keep_last_returns'}))

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 0
    assert stayers.iloc[0]['j'] == 0
    assert stayers.iloc[0]['y'] == 1
    assert stayers.iloc[0]['t'] == 6

    assert movers.iloc[0]['i'] == 0
    assert movers.iloc[0]['j'] == 1
    assert movers.iloc[0]['y'] == 1
    assert movers.iloc[0]['t'] == 3

    assert movers.iloc[1]['i'] == 0
    assert movers.iloc[1]['j'] == 0
    assert movers.iloc[1]['y'] == 1
    assert movers.iloc[1]['t'] == 5

    assert movers.iloc[2]['i'] == 1
    assert movers.iloc[2]['j'] == 1
    assert movers.iloc[2]['y'] == 1
    assert movers.iloc[2]['t'] == 1

    assert movers.iloc[3]['i'] == 1
    assert movers.iloc[3]['j'] == 2
    assert movers.iloc[3]['y'] == 1
    assert movers.iloc[3]['t'] == 2

    assert movers.iloc[4]['i'] == 2
    assert movers.iloc[4]['j'] == 2
    assert movers.iloc[4]['y'] == 1
    assert movers.iloc[4]['t'] == 1

    assert movers.iloc[5]['i'] == 2
    assert movers.iloc[5]['j'] == 1
    assert movers.iloc[5]['y'] == 2
    assert movers.iloc[5]['t'] == 2

def test_min_obs_ids_28_1():
    # List only firms that meet a minimum threshold of observations.
    # Using long/event study.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()

    threshold = 250

    # First, manually estimate the valid set of firms
    frame = bdf.copy()
    n_moves = frame.groupby('j')['i'].size()

    valid_firms = np.sort(n_moves[n_moves >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = np.sort(bdf.min_obs_ids(threshold=threshold, id_col='j'))
    valid_firms3 = np.sort(bdf.to_eventstudy().min_obs_ids(threshold=threshold, id_col='j'))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3)
    assert np.all(valid_firms == valid_firms2)
    assert np.all(valid_firms == valid_firms3)

def test_min_obs_ids_28_2():
    # List only firms that meet a minimum threshold of observations.
    # Using long collapsed/event study collapsed.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean().collapse()

    threshold = 60

    # First, manually estimate the valid set of firms
    frame = bdf.copy()
    n_moves = frame.groupby('j')['i'].size()

    valid_firms = np.sort(n_moves[n_moves >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = np.sort(bdf.min_obs_ids(threshold=threshold, id_col='j'))
    valid_firms3 = np.sort(bdf.to_eventstudy().min_obs_ids(threshold=threshold, id_col='j'))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3)
    assert np.all(valid_firms == valid_firms2)
    assert np.all(valid_firms == valid_firms3)

def test_min_obs_frame_29_1():
    # Keep only firms that meet a minimum threshold of observations.
    # Using long/event study.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()

    threshold = 250

    # First, manually estimate the new frame
    frame = bdf.copy()
    n_moves = frame.groupby('j')['i'].size()

    valid_firms = sorted(n_moves[n_moves >= threshold].index)
    new_frame = frame.keep_ids('j', valid_firms)
    new_frame.reset_index(drop=True, inplace=True)

    # Next, estimate the new frame using the built-in function
    new_frame2 = bdf.min_obs_frame(threshold=threshold, id_col='j')
    new_frame3 = bdf.to_eventstudy().min_obs_frame(threshold=threshold, id_col='j').to_long()

    assert (0 < len(new_frame) < len(bdf))
    assert len(new_frame) == len(new_frame2) == len(new_frame3)
    for col in ['i', 'j', 'y', 't']:
        # Skip 'm' since we didn't recompute it
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame2.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame3.loc[:, col].to_numpy())

def test_min_obs_frame_29_2():
    # Keep only firms that meet a minimum threshold of observations.
    # Using long collapsed/event study collapsed.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean().collapse()

    threshold = 60

    # First, manually estimate the new frame
    frame = bdf.copy()
    n_moves = frame.groupby('j')['i'].size()

    valid_firms = sorted(n_moves[n_moves >= threshold].index)
    new_frame = frame.keep_ids('j', valid_firms)

    # Next, estimate the new frame using the built-in function
    new_frame2 = bdf.min_obs_frame(threshold=threshold, id_col='j')
    new_frame3 = bdf.to_eventstudy().min_obs_frame(threshold=threshold, id_col='j').to_long()

    assert (0 < len(new_frame) < len(bdf))
    assert len(new_frame) == len(new_frame2) == len(new_frame3)
    for col in ['i', 'j', 'y', 't1', 't2']:
        # Skip 'm' since we didn't recompute it
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame2.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame3.loc[:, col].to_numpy())

def test_min_joint_obs_frame_29_3():
    # Keep only firms that meet a minimum threshold of observations and workers that meet a separate minimum threshold of observations.
    # Using long/event study/long collapsed/event study collapsed.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean(bpd.clean_params({'verbose': False}))

    threshold_1 = 10
    threshold_2 = 2

    # Estimate the new frame using the built-in function
    new_frame2 = bdf.min_joint_obs_frame(threshold_1, threshold_2)
    new_frame3 = bdf.to_eventstudy().min_joint_obs_frame(threshold_1, threshold_2).to_long()
    new_frame4 = bdf.collapse().min_joint_obs_frame(threshold_1, threshold_2)
    new_frame5 = bdf.collapse().min_joint_obs_frame(threshold_2, threshold_1, 'i', 'j')
    new_frame6 = bdf.collapse().to_eventstudy().min_joint_obs_frame(threshold_1, threshold_2).to_long()
    new_frame7 = bdf.collapse().to_eventstudy().min_joint_obs_frame(threshold_2, threshold_1, 'i', 'j').to_long()

    assert (0 < len(new_frame4) < len(new_frame2) <= len(bdf))
    assert len(new_frame2) == len(new_frame3)
    assert len(new_frame4) == len(new_frame5) == len(new_frame6) == len(new_frame7)
    assert np.min(new_frame2.groupby('j').size()) >= threshold_1
    assert np.min(new_frame2.groupby('i').size()) >= threshold_2
    assert np.min(new_frame4.groupby('j').size()) >= threshold_1
    assert np.min(new_frame4.groupby('i').size()) >= threshold_2
    for col in ['i', 'j', 'y', 't']:
        # Skip 'm' since we didn't recompute it
        assert np.all(new_frame2.loc[:, col].to_numpy() == new_frame3.loc[:, col].to_numpy())
    for col in ['i', 'j', 'y', 't1', 't2']:
        # Skip 'm' since we didn't recompute it
        assert np.all(new_frame4.loc[:, col].to_numpy() == new_frame5.loc[:, col].to_numpy())
        assert np.all(new_frame4.loc[:, col].to_numpy() == new_frame6.loc[:, col].to_numpy())
        assert np.all(new_frame4.loc[:, col].to_numpy() == new_frame7.loc[:, col].to_numpy())

def test_min_workers_firms_30():
    # List only firms that meet a minimum threshold of workers.
    # Using long/event study/long collapsed/event study collapsed.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()

    threshold = 40

    # First, manually estimate the valid set of firms
    frame = bdf.copy()
    # Count workers
    n_workers = frame.groupby('j')['i'].nunique()

    valid_firms = np.sort(n_workers[n_workers >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = np.sort(bdf.min_workers_firms(threshold))
    valid_firms3 = np.sort(bdf.to_eventstudy().min_workers_firms(threshold))
    valid_firms4 = np.sort(bdf.collapse().min_workers_firms(threshold))
    valid_firms5 = np.sort(bdf.collapse().to_eventstudy().min_workers_firms(threshold))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3) == len(valid_firms4) == len(valid_firms5)
    assert np.all(valid_firms == valid_firms2)
    assert np.all(valid_firms == valid_firms3)
    assert np.all(valid_firms == valid_firms4)
    assert np.all(valid_firms == valid_firms5)

def test_min_workers_frame_31():
    # Keep only firms that meet a minimum threshold of workers.
    # Using long/event study/long collapsed/event study collapsed.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()

    threshold = 60

    # First, manually estimate the new frame
    frame = bdf.copy()
    # Count workers
    n_workers = frame.groupby('j')['i'].nunique()

    valid_firms = n_workers[n_workers >= threshold].index
    new_frame = frame.keep_ids('j', valid_firms).collapse()

    # Next, estimate the new frame using the built-in function
    new_frame2 = bdf.min_workers_frame(threshold).collapse()
    new_frame3 = bdf.to_eventstudy().min_workers_frame(threshold).to_long().collapse()
    new_frame4 = bdf.collapse().min_workers_frame(threshold)
    new_frame5 = bdf.collapse().to_eventstudy().min_workers_frame(threshold).to_long()

    assert (0 < len(new_frame) < len(bdf))
    assert len(new_frame) == len(new_frame2) == len(new_frame3) == len(new_frame4) == len(new_frame5)
    for col in ['i', 'j', 'y', 't1', 't2']:
        # Skip 'm' since we didn't recompute it
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame2.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame3.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame4.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame5.loc[:, col].to_numpy())

def test_min_moves_firms_32_1():
    # List only firms that meet a minimum threshold of moves.
    # Using long/event study.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()

    threshold = 20

    # First, manually estimate the valid set of firms
    frame = bdf.copy()
    frame.loc[frame.loc[:, 'm'] == 2, 'm'] = 1
    n_moves = frame.groupby('j')['m'].sum()

    valid_firms = np.sort(n_moves[n_moves >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = np.sort(bdf.min_moves_firms(threshold))
    valid_firms3 = np.sort(bdf.to_eventstudy().min_moves_firms(threshold))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3)
    assert np.all(valid_firms == valid_firms2)
    assert np.all(valid_firms == valid_firms3)

def test_min_moves_firms_32_2():
    # List only firms that meet a minimum threshold of moves.
    # Using long collapsed/event study collapsed.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean().collapse()

    threshold = 20

    # First, manually estimate the valid set of firms
    frame = bdf.copy()
    frame.loc[frame.loc[:, 'm'] == 2, 'm'] = 1
    n_moves = frame.groupby('j')['m'].sum()

    valid_firms = np.sort(n_moves[n_moves >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = np.sort(bdf.min_moves_firms(threshold))
    valid_firms3 = np.sort(bdf.to_eventstudy().min_moves_firms(threshold))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3)
    assert np.all(valid_firms == valid_firms2)
    assert np.all(valid_firms == valid_firms3)

def test_min_moves_frame_33():
    # Keep only firms that meet a minimum threshold of moves.
    # Using long/event study/long collapsed/event study collapsed.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()

    threshold = 12

    # First, manually estimate the valid set of firms
    frame = bdf.copy()
    frame.loc[frame.loc[:, 'm'] == 2, 'm'] = 1
    n_moves = frame.groupby('j')['m'].sum()

    valid_firms = sorted(n_moves[n_moves >= threshold].index)
    new_frame = frame.keep_ids('j', valid_firms)

    # Iterate until set of firms stays the same between loops
    loop = True
    n_loops = 0
    while loop:
        n_loops += 1
        prev_frame = new_frame
        prev_frame.loc[prev_frame.loc[:, 'm'] == 2, 'm'] = 1
        # Keep firms with sufficiently many moves
        n_moves = prev_frame.groupby('j')['m'].sum()
        valid_firms = sorted(n_moves[n_moves >= threshold].index)
        new_frame = prev_frame.keep_ids('j', valid_firms)
        loop = (len(new_frame) != len(prev_frame))
    new_frame = new_frame.collapse()

    # Next, estimate the new frame using the built-in function
    new_frame2 = bdf.min_moves_frame(threshold).collapse()
    new_frame3 = bdf.to_eventstudy().min_moves_frame(threshold).to_long().collapse()
    new_frame4 = bdf.collapse().min_moves_frame(threshold)
    new_frame5 = bdf.collapse().to_eventstudy().min_moves_frame(threshold).to_long()

    assert n_loops > 1
    assert (0 < len(new_frame) < len(bdf))
    assert len(new_frame) == len(new_frame2) == len(new_frame3) == len(new_frame4) == len(new_frame5)
    for col in ['i', 'j', 'y', 't1', 't2']:
        # Skip 'm' since we didn't recompute it
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame2.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame3.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame4.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame5.loc[:, col].to_numpy())

def test_min_movers_firms_34():
    # List only firms that meet a minimum threshold of movers.
    # Using long/event study/long collapsed/event study collapsed.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()

    threshold = 20

    # First, manually estimate the valid set of firms
    frame = bdf.copy()
    # Keep movers
    frame = frame[frame['m'] > 0]
    n_movers = frame.groupby('j')['i'].nunique()

    valid_firms = np.sort(n_movers[n_movers >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = np.sort(bdf.min_movers_firms(threshold))
    valid_firms3 = np.sort(bdf.to_eventstudy().min_movers_firms(threshold))
    valid_firms4 = np.sort(bdf.collapse().min_movers_firms(threshold))
    valid_firms5 = np.sort(bdf.collapse().to_eventstudy().min_movers_firms(threshold))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3) == len(valid_firms4) == len(valid_firms5)
    assert np.all(valid_firms == valid_firms2)
    assert np.all(valid_firms == valid_firms3)
    assert np.all(valid_firms == valid_firms4)
    assert np.all(valid_firms == valid_firms5)

def test_min_movers_frame_35():
    # Keep only firms that meet a minimum threshold of movers.
    # Using long/event study/long collapsed/event study collapsed.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()

    threshold = 12

    # First, manually estimate the new frame
    frame = bdf.copy()
    # Keep movers
    frame_movers = frame[frame['m'] > 0]
    n_movers = frame_movers.groupby('j')['i'].nunique()

    valid_firms = n_movers[n_movers >= threshold].index
    new_frame = frame.keep_ids('j', valid_firms)

    # Iterate until set of firms stays the same between loops
    loop = True
    n_loops = 0
    while loop:
        n_loops += 1
        prev_frame = new_frame
        # Keep movers
        prev_frame_movers = prev_frame[prev_frame['m'] > 0]
        n_movers = prev_frame_movers.groupby('j')['i'].nunique()
        valid_firms = n_movers[n_movers >= threshold].index
        new_frame = prev_frame.keep_ids('j', valid_firms)
        loop = (len(new_frame) != len(prev_frame))
    new_frame = new_frame.collapse()

    # Next, estimate the new frame using the built-in function
    new_frame2 = bdf.min_movers_frame(threshold).collapse()
    new_frame3 = bdf.to_eventstudy().min_movers_frame(threshold).to_long().collapse()
    new_frame4 = bdf.collapse().min_movers_frame(threshold)
    new_frame5 = bdf.collapse().to_eventstudy().min_movers_frame(threshold).to_long()

    assert n_loops > 1
    assert (0 < len(new_frame) < len(bdf))
    assert len(new_frame) == len(new_frame2) == len(new_frame3) == len(new_frame4) == len(new_frame5)
    for col in ['i', 'j', 'y', 't1', 't2']:
        # Skip 'm' since we didn't recompute it
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame2.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame3.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame4.loc[:, col].to_numpy())
        assert np.all(new_frame.loc[:, col].to_numpy() == new_frame5.loc[:, col].to_numpy())

def test_construct_artificial_time_36():
    # Test construct_artificial_time() methods
    # First, on non-collapsed data
    a = bpd.BipartiteLong(bpd.SimBipartite().simulate(np.random.default_rng(1234))[['i', 'j', 'y', 't']]).clean()
    b = a.drop('t', axis=1, inplace=False, allow_optional=True).construct_artificial_time(copy=True).to_eventstudy().drop('t', axis=1, inplace=False, allow_optional=True).construct_artificial_time(time_per_worker=True, is_sorted=True, copy=False).to_long()

    assert np.all(a.to_numpy() == b.to_numpy())

    # Second, on collapsed data
    a = a.collapse()
    b = a.drop('t', axis=1, inplace=False, allow_optional=True).construct_artificial_time(copy=True).to_eventstudy().drop('t', axis=1, inplace=False, allow_optional=True).construct_artificial_time(time_per_worker=True, is_sorted=True, copy=False).to_long()

    assert np.all(a[['i', 'j', 'y', 'm']].to_numpy() == b[['i', 'j', 'y', 'm']].to_numpy())
