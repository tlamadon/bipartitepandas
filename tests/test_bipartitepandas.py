'''
Tests for bipartitepandas

DATE: March 2021
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
    # Check contiguous_ids() with firm ids.
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
    # Check contiguous_ids() with worker ids.
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
    # Check contiguous_ids() with cluster ids.
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
    # Check contiguous_ids() with cluster ids.
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

    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True)
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
        bdf = bpd.BipartiteEventStudy(data=df.copy(), include_id_reference_dict=True)
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

    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True)
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

    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True)
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

    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True)
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

def test_min_obs_firms_28_1():
    # List only firms that meet a minimum threshold of observations.
    # Using long/event study.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean()

    threshold = 250

    # First, manually estimate the valid set of firms
    frame = bdf.copy()
    n_moves = frame.groupby('j')['i'].size()

    valid_firms = sorted(n_moves[n_moves >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = sorted(bdf.min_obs_firms(threshold))
    valid_firms3 = sorted(bdf.to_eventstudy().min_obs_firms(threshold))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3)
    for i in range(len(valid_firms)):
        assert valid_firms[i] == valid_firms2[i] == valid_firms3[i]

def test_min_obs_firms_28_2():
    # List only firms that meet a minimum threshold of observations.
    # Using long collapsed/event study collapsed.
    df = bpd.SimBipartite(bpd.sim_params({'p_move': 0.05})).simulate(np.random.default_rng(1234))
    bdf = bpd.BipartiteLong(df[['i', 'j', 'y', 't']]).clean().collapse()

    threshold = 60

    # First, manually estimate the valid set of firms
    frame = bdf.copy()
    n_moves = frame.groupby('j')['i'].size()

    valid_firms = sorted(n_moves[n_moves >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = sorted(bdf.min_obs_firms(threshold))
    valid_firms3 = sorted(bdf.to_eventstudy().min_obs_firms(threshold))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3)
    for i in range(len(valid_firms)):
        assert valid_firms[i] == valid_firms2[i] == valid_firms3[i]

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
    new_frame2 = bdf.min_obs_frame(threshold)
    new_frame3 = bdf.to_eventstudy().min_obs_frame(threshold).to_long()

    assert (0 < len(new_frame) < len(bdf))
    assert len(new_frame) == len(new_frame2) == len(new_frame3)
    for i in range(100): # range(len(new_frame)): # It takes too long to go through all rows
        for col in ['i', 'j', 'y', 't']:
            # Skip 'm' since we didn't recompute it
            assert new_frame.iloc[i][col] == new_frame2.iloc[i][col] == new_frame3.iloc[i][col]
    for i in range(len(new_frame) - 100, len(new_frame)): # range(len(new_frame)): # It takes too long to go through all rows
        for col in ['i', 'j', 'y', 't']:
            # Skip 'm' since we didn't recompute it
            assert new_frame.iloc[i][col] == new_frame2.iloc[i][col] == new_frame3.iloc[i][col]

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
    new_frame2 = bdf.min_obs_frame(threshold)
    new_frame3 = bdf.to_eventstudy().min_obs_frame(threshold).to_long()

    assert (0 < len(new_frame) < len(bdf))
    assert len(new_frame) == len(new_frame2) == len(new_frame3)
    for i in range(100): # range(len(new_frame)): # It takes too long to go through all rows
        for col in ['i', 'j', 'y', 't1', 't2']:
            # Skip 'm' since we didn't recompute it
            assert new_frame.iloc[i][col] == new_frame2.iloc[i][col] == new_frame3.iloc[i][col]
    for i in range(len(new_frame) - 100, len(new_frame)): # range(len(new_frame)): # It takes too long to go through all rows
        for col in ['i', 'j', 'y', 't1', 't2']:
            # Skip 'm' since we didn't recompute it
            assert new_frame.iloc[i][col] == new_frame2.iloc[i][col] == new_frame3.iloc[i][col]

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

    valid_firms = sorted(n_workers[n_workers >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = sorted(bdf.min_workers_firms(threshold))
    valid_firms3 = sorted(bdf.to_eventstudy().min_workers_firms(threshold))
    valid_firms4 = sorted(bdf.collapse().min_workers_firms(threshold))
    valid_firms5 = sorted(bdf.collapse().to_eventstudy().min_workers_firms(threshold))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3) == len(valid_firms4) == len(valid_firms5)
    for i in range(len(valid_firms)):
        assert valid_firms[i] == valid_firms2[i] == valid_firms3[i] == valid_firms4[i] == valid_firms5[i]

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
    for i in range(100): # range(len(new_frame)): # It takes too long to go through all rows
        for col in ['i', 'j', 'y', 't1', 't2']:
            # Skip 'm' since we didn't recompute it
            assert new_frame.iloc[i][col] == new_frame2.iloc[i][col] == new_frame3.iloc[i][col] == new_frame4.iloc[i][col] == new_frame5.iloc[i][col]
    for i in range(len(new_frame) - 100, len(new_frame)): # range(len(new_frame)): # It takes too long to go through all rows
        for col in ['i', 'j', 'y', 't1', 't2']:
            # Skip 'm' since we didn't recompute it
            assert new_frame.iloc[i][col] == new_frame2.iloc[i][col] == new_frame3.iloc[i][col] == new_frame4.iloc[i][col] == new_frame5.iloc[i][col]

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

    valid_firms = sorted(n_moves[n_moves >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = sorted(bdf.min_moves_firms(threshold))
    valid_firms3 = sorted(bdf.to_eventstudy().min_moves_firms(threshold))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3)
    for i in range(len(valid_firms)):
        assert valid_firms[i] == valid_firms2[i] == valid_firms3[i]

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

    valid_firms = sorted(n_moves[n_moves >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = sorted(bdf.min_moves_firms(threshold))
    valid_firms3 = sorted(bdf.to_eventstudy().min_moves_firms(threshold))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3)
    for i in range(len(valid_firms)):
        assert valid_firms[i] == valid_firms2[i] == valid_firms3[i]

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
    for i in range(100): # range(len(new_frame)): # It takes too long to go through all rows
        for col in ['i', 'j', 'y', 't1', 't2']:
            # Skip 'm' since we didn't recompute it
            assert new_frame.iloc[i][col] == new_frame2.iloc[i][col] == new_frame3.iloc[i][col] == new_frame4.iloc[i][col] == new_frame5.iloc[i][col]
    for i in range(len(new_frame) - 100, len(new_frame)): # range(len(new_frame)): # It takes too long to go through all rows
        for col in ['i', 'j', 'y', 't1', 't2']:
            # Skip 'm' since we didn't recompute it
            assert new_frame.iloc[i][col] == new_frame2.iloc[i][col] == new_frame3.iloc[i][col] == new_frame4.iloc[i][col] == new_frame5.iloc[i][col]

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

    valid_firms = sorted(n_movers[n_movers >= threshold].index)

    # Next, estimate the set of valid firms using the built-in function
    valid_firms2 = sorted(bdf.min_movers_firms(threshold))
    valid_firms3 = sorted(bdf.to_eventstudy().min_movers_firms(threshold))
    valid_firms4 = sorted(bdf.collapse().min_movers_firms(threshold))
    valid_firms5 = sorted(bdf.collapse().to_eventstudy().min_movers_firms(threshold))

    assert (0 < len(valid_firms) < df['j'].nunique())
    assert len(valid_firms) == len(valid_firms2) == len(valid_firms3) == len(valid_firms4) == len(valid_firms5)
    for i in range(len(valid_firms)):
        assert valid_firms[i] == valid_firms2[i] == valid_firms3[i] == valid_firms4[i] == valid_firms5[i]

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
    for i in range(100): # range(len(new_frame)): # It takes too long to go through all rows
        for col in ['i', 'j', 'y', 't1', 't2']:
            # Skip 'm' since we didn't recompute it
            assert new_frame.iloc[i][col] == new_frame2.iloc[i][col] == new_frame3.iloc[i][col] == new_frame4.iloc[i][col] == new_frame5.iloc[i][col]
    for i in range(len(new_frame) - 100, len(new_frame)): # range(len(new_frame)): # It takes too long to go through all rows
        for col in ['i', 'j', 'y', 't1', 't2']:
            # Skip 'm' since we didn't recompute it
            assert new_frame.iloc[i][col] == new_frame2.iloc[i][col] == new_frame3.iloc[i][col] == new_frame4.iloc[i][col] == new_frame5.iloc[i][col]

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

###################################
##### Tests for BipartiteLong #####
###################################

def test_long_get_extended_eventstudy_1():
    # Test get_extended_eventstudy() by making sure it is generating the event study correctly for periods_pre=2 and periods_post=1
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 3})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 2., 't': 3})
    worker_data.append({'i': 1, 'j': 5, 'y': 1., 't': 3})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2})
    worker_data.append({'i': 3, 'j': 3, 'y': 1.5, 't': 3})
    # Worker 4
    worker_data.append({'i': 4, 'j': 0, 'y': 1., 't': 1})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])
    df['g'] = df['j'] # Fill in g column as j

    bdf = bpd.BipartiteLong(df)
    bdf = bdf.clean()

    es_extended = bdf.get_extended_eventstudy(transition_col='g', outcomes=['j', 'y'], periods_pre=2, periods_post=1)

    assert es_extended.iloc[0]['i'] == 0
    assert es_extended.iloc[0]['j_l2'] == 1
    assert es_extended.iloc[0]['j_l1'] == 1
    assert es_extended.iloc[0]['j_f1'] == 0
    assert es_extended.iloc[0]['y_l2'] == 1
    assert es_extended.iloc[0]['y_l1'] == 1
    assert es_extended.iloc[0]['y_f1'] == 1
    assert es_extended.iloc[0]['t'] == 4

    assert es_extended.iloc[1]['i'] == 2
    assert es_extended.iloc[1]['j_l2'] == 2
    assert es_extended.iloc[1]['j_l1'] == 2
    assert es_extended.iloc[1]['j_f1'] == 3
    assert es_extended.iloc[1]['y_l2'] == 1
    assert es_extended.iloc[1]['y_l1'] == 1
    assert es_extended.iloc[1]['y_f1'] == 1.5
    assert es_extended.iloc[1]['t'] == 3

def test_long_get_extended_eventstudy_2():
    # Test get_extended_eventstudy() by making sure workers move firms at the fulcrum of the event study
    sim_data = bpd.SimBipartite().simulate()
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()

    es_extended = bdf.get_extended_eventstudy(outcomes=['j', 'y'], periods_pre=3, periods_post=2)

    assert np.sum(es_extended['j_l1'] == es_extended['j_f1']) == 0

def test_long_get_extended_eventstudy_3_1():
    # Test get_extended_eventstudy() by making sure workers move firms at the fulcrum of the event study and stable_pre works
    sim_data = bpd.SimBipartite().simulate()
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()

    es_extended = bdf.get_extended_eventstudy(outcomes=['j', 'y'], periods_pre=2, periods_post=3, stable_pre='j')

    assert np.sum(es_extended['j_l2'] != es_extended['j_l1']) == 0
    assert np.sum(es_extended['j_l1'] == es_extended['j_f1']) == 0

def test_long_get_extended_eventstudy_3_2():
    # Test get_extended_eventstudy() by making sure workers move firms at the fulcrum of the event study and stable_post works
    sim_data = bpd.SimBipartite().simulate()
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()

    es_extended = bdf.get_extended_eventstudy(outcomes=['j', 'y'], periods_pre=3, periods_post=2, stable_post='j')

    assert np.sum(es_extended['j_l1'] == es_extended['j_f1']) == 0
    assert np.sum(es_extended['j_f1'] != es_extended['j_f2']) == 0

def test_long_get_extended_eventstudy_3_3():
    # Test get_extended_eventstudy() by making sure workers move firms at the fulcrum of the event study and stable_post and stable_pre work together
    sim_data = bpd.SimBipartite().simulate()
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()

    es_extended = bdf.get_extended_eventstudy(outcomes=['j', 'y'], periods_pre=3, periods_post=2, stable_pre='j', stable_post='j')

    assert len(es_extended) > 0 # Make sure something is left
    assert np.sum(es_extended['j_l3'] != es_extended['j_l2']) == 0
    assert np.sum(es_extended['j_l2'] != es_extended['j_l1']) == 0
    assert np.sum(es_extended['j_l1'] == es_extended['j_f1']) == 0
    assert np.sum(es_extended['j_f1'] != es_extended['j_f2']) == 0

# Only uncomment for manual testing - this produces a graph which pauses the testing
# def test_long_plot_extended_eventstudy_4():
#     # Test plot_extended_eventstudy() by making sure it doesn't crash
#     sim_data = bpd.SimBipartite().simulate()
#     bdf = bpd.BipartiteLong(sim_data).clean().cluster(grouping=bpd.grouping.KMeans(n_clusters=2))
#     bdf.plot_extended_eventstudy()

#     assert True # Just making sure it doesn't crash

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

#########################################
##### Tests for BipartiteEventStudy #####
#########################################

def test_event_study_1():
    # Test constructor for BipartiteEventStudy.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    df = pd.DataFrame(bdf.to_eventstudy())
    bdf = bpd.BipartiteEventStudy(df)
    bdf = bdf.clean()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

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

def test_get_cs_2():
    # Test get_cs() for BipartiteEventStudy.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 2., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    df = pd.DataFrame(bdf.to_eventstudy()).rename({'t1': 't'}, axis=1)
    bdf = bpd.BipartiteEventStudy(df, col_dict={'t1': 't'})
    bdf = bdf.clean()
    bdf = bdf.get_cs(copy=False)

    stayers = bdf[bdf['m'] == 0]
    movers1 = bdf[(bdf['m'] > 0) & (bdf['cs'] == 1)]
    movers0 = bdf[(bdf['m'] > 0) & (bdf['cs'] == 0)]

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

##################################################
##### Tests for BipartiteEventStudyCollapsed #####
##################################################

def test_event_study_collapsed_1():
    # Test constructor for BipartiteEventStudyCollapsed.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 2., 't': 3})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    df = pd.DataFrame(bdf.to_eventstudy())
    bdf = bpd.BipartiteEventStudyCollapsed(df)
    bdf = bdf.clean()

    stayers = bdf[bdf['m'] == 0]
    movers = bdf[bdf['m'] > 0]

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['t11'] == 1
    assert stayers.iloc[0]['t12'] == 2
    assert stayers.iloc[0]['t21'] == 1
    assert stayers.iloc[0]['t22'] == 2

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

def test_get_cs_2():
    # Test get_cs() for BipartiteEventStudyCollapsed.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 2., 't': 3})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()
    bdf = bdf.collapse()
    df = pd.DataFrame(bdf.to_eventstudy())
    bdf = bpd.BipartiteEventStudyCollapsed(df)
    bdf = bdf.clean()
    bdf = bdf.get_cs(copy=False)

    stayers = bdf[bdf['m'] == 0]
    movers1 = bdf[(bdf['m'] > 0) & (bdf['cs'] == 1)]
    movers0 = bdf[(bdf['m'] > 0) & (bdf['cs'] == 0)]

    assert stayers.iloc[0]['i'] == 2
    assert stayers.iloc[0]['j1'] == 2
    assert stayers.iloc[0]['j2'] == 2
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1
    assert stayers.iloc[0]['t11'] == 1
    assert stayers.iloc[0]['t12'] == 2
    assert stayers.iloc[0]['t21'] == 1
    assert stayers.iloc[0]['t22'] == 2

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

###################################
##### Tests for Connectedness #####
###################################

def test_connectedness_1():
    # Test connected and leave-one-firm-out for collapsed long format.
    # There are 2 biconnected sets that are connected by 1 mover, so the largest connected set is all the observations, while the largest biconnected component is the larger of the 2 biconnected sets.
    worker_data = []
    # Group 1 is firms 0 to 5
    # Worker 0
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 1, 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 2})
    # Worker 1
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 2, 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1, 't': 2})
    # Worker 2
    # Firm 2 -> 3
    worker_data.append({'i': 2, 'j': 2, 'y': 3, 't': 1})
    worker_data.append({'i': 2, 'j': 3, 'y': 2.5, 't': 2})
    # Worker 3
    # Firm 3 -> 4
    worker_data.append({'i': 3, 'j': 3, 'y': 1, 't': 1})
    worker_data.append({'i': 3, 'j': 4, 'y': 1, 't': 2})
    # Worker 4
    # Firm 4 -> 5
    worker_data.append({'i': 4, 'j': 4, 'y': 1, 't': 1})
    worker_data.append({'i': 4, 'j': 5, 'y': 1.3, 't': 2})
    # Worker 5
    # Firm 5 -> 0
    worker_data.append({'i': 5, 'j': 5, 'y': 1.1, 't': 1})
    worker_data.append({'i': 5, 'j': 0, 'y': 1, 't': 2})
    # Group 2 is firms 6 to 9
    # Group 2 is linked to Group 1 through firms 3 and 6
    # Worker 6
    # Firm 3 -> 6
    worker_data.append({'i': 6, 'j': 3, 'y': 2, 't': 1})
    worker_data.append({'i': 6, 'j': 6, 'y': 1, 't': 2})
    # Worker 7
    # Firm 6 -> 7
    worker_data.append({'i': 7, 'j': 6, 'y': 1, 't': 1})
    worker_data.append({'i': 7, 'j': 7, 'y': 2, 't': 2})
    # Worker 8
    # Firm 7 -> 8
    worker_data.append({'i': 8, 'j': 7, 'y': 1.5, 't': 1})
    worker_data.append({'i': 8, 'j': 8, 'y': 1.2, 't': 2})
    # Worker 9
    # Firm 8 -> 9
    worker_data.append({'i': 9, 'j': 8, 'y': 1.6, 't': 1})
    worker_data.append({'i': 9, 'j': 9, 'y': 2, 't': 2})
    # Worker 10
    # Firm 9 -> 6
    worker_data.append({'i': 10, 'j': 9, 'y': 1.8, 't': 1})
    worker_data.append({'i': 10, 'j': 6, 'y': 1.4, 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    # Connected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'connected'}))
    assert bdf.n_firms() == 10

    # Biconnected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_firm'}))
    assert bdf.n_firms() == 6

def test_connectedness_2():
    # Test connected and leave-one-firm-out for collapsed long format. Now, add two firms connected to the largest biconnected set that are linked only by 1 mover.
    # There are 3 biconnected sets that are connected by 1 mover, so the largest connected set is all the observations, while the largest biconnected component is the larger of the 3 biconnected sets.
    worker_data = []
    # Group 1 is firms 0 to 5
    # Worker 0
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 1, 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 2})
    # Worker 1
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 2, 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1, 't': 2})
    # Worker 2
    # Firm 2 -> 3
    worker_data.append({'i': 2, 'j': 2, 'y': 3, 't': 1})
    worker_data.append({'i': 2, 'j': 3, 'y': 2.5, 't': 2})
    # Worker 3
    # Firm 3 -> 4
    worker_data.append({'i': 3, 'j': 3, 'y': 1, 't': 1})
    worker_data.append({'i': 3, 'j': 4, 'y': 1, 't': 2})
    # Worker 4
    # Firm 4 -> 5
    worker_data.append({'i': 4, 'j': 4, 'y': 1, 't': 1})
    worker_data.append({'i': 4, 'j': 5, 'y': 1.3, 't': 2})
    # Worker 5
    # Firm 5 -> 0
    worker_data.append({'i': 5, 'j': 5, 'y': 1.1, 't': 1})
    worker_data.append({'i': 5, 'j': 0, 'y': 1, 't': 2})
    # Group 2 is firms 6 to 9
    # Group 2 is linked to Group 1 through firms 3 and 6
    # Worker 6
    # Firm 3 -> 6
    worker_data.append({'i': 6, 'j': 3, 'y': 2, 't': 1})
    worker_data.append({'i': 6, 'j': 6, 'y': 1, 't': 2})
    # Worker 7
    # Firm 6 -> 7
    worker_data.append({'i': 7, 'j': 6, 'y': 1, 't': 1})
    worker_data.append({'i': 7, 'j': 7, 'y': 2, 't': 2})
    # Worker 8
    # Firm 7 -> 8
    worker_data.append({'i': 8, 'j': 7, 'y': 1.5, 't': 1})
    worker_data.append({'i': 8, 'j': 8, 'y': 1.2, 't': 2})
    # Worker 9
    # Firm 8 -> 9
    worker_data.append({'i': 9, 'j': 8, 'y': 1.6, 't': 1})
    worker_data.append({'i': 9, 'j': 9, 'y': 2, 't': 2})
    # Worker 10
    # Firm 9 -> 6
    worker_data.append({'i': 10, 'j': 9, 'y': 1.8, 't': 1})
    worker_data.append({'i': 10, 'j': 6, 'y': 1.4, 't': 2})
    # Group 3 is firms 10 to 11
    # Worker 11
    # Firm 10 -> 4
    worker_data.append({'i': 11, 'j': 10, 'y': 1.3, 't': 1})
    worker_data.append({'i': 11, 'j': 4, 'y': 1.2, 't': 2})
    worker_data.append({'i': 11, 'j': 11, 'y': 1, 't': 3})
    worker_data.append({'i': 11, 'j': 10, 'y': 1.1, 't': 4})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    # Connected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'connected'}))
    assert bdf.n_firms() == 12

    # Biconnected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_firm'}))
    assert bdf.n_firms() == 6

def test_connectedness_3():
    # Test connected and leave-one-firm-out for collapsed long format.
    # There are 2 biconnected sets that are connected by 1 mover, so the largest connected set is all the observations. However, unlike test 1, the largest biconnected component is also all the observations. This is because individual 2 goes from firm 2 to 3 to 6 to 7. It seems that removing 3 disconnects the 2 groups, but the shifts used to compute the biconnected components corrects this.
    worker_data = []
    # Group 1 is firms 0 to 5
    # Worker 0
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 1, 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 2})
    # Worker 1
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 2, 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1, 't': 2})
    # Worker 2
    # Firm 2 -> 3 -> 6 -> 7
    worker_data.append({'i': 2, 'j': 2, 'y': 3, 't': 1})
    worker_data.append({'i': 2, 'j': 3, 'y': 2.5, 't': 2})
    worker_data.append({'i': 2, 'j': 6, 'y': 1, 't': 3})
    worker_data.append({'i': 2, 'j': 7, 'y': 2.5, 't': 4})
    # Worker 3
    # Firm 3 -> 4
    worker_data.append({'i': 3, 'j': 3, 'y': 1, 't': 1})
    worker_data.append({'i': 3, 'j': 4, 'y': 1, 't': 2})
    # Worker 4
    # Firm 4 -> 5
    worker_data.append({'i': 4, 'j': 4, 'y': 1, 't': 1})
    worker_data.append({'i': 4, 'j': 5, 'y': 1.3, 't': 2})
    # Worker 5
    # Firm 5 -> 0
    worker_data.append({'i': 5, 'j': 5, 'y': 1.1, 't': 1})
    worker_data.append({'i': 5, 'j': 0, 'y': 1, 't': 2})
    # Group 2 is firms 6 to 9
    # Group 2 is linked to Group 1 through firms 2, 3, 6, and 7
    # Worker 6
    # Firm 6 -> 7
    worker_data.append({'i': 6, 'j': 6, 'y': 1, 't': 1})
    worker_data.append({'i': 6, 'j': 7, 'y': 2, 't': 2})
    # Worker 7
    # Firm 7 -> 8
    worker_data.append({'i': 7, 'j': 7, 'y': 1.5, 't': 1})
    worker_data.append({'i': 7, 'j': 8, 'y': 1.2, 't': 2})
    # Worker 8
    # Firm 8 -> 9
    worker_data.append({'i': 8, 'j': 8, 'y': 1.6, 't': 1})
    worker_data.append({'i': 8, 'j': 9, 'y': 2, 't': 2})
    # Worker 9
    # Firm 9 -> 6
    worker_data.append({'i': 9, 'j': 9, 'y': 1.8, 't': 1})
    worker_data.append({'i': 9, 'j': 6, 'y': 1.4, 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    # Connected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'connected'}))
    assert bdf.n_firms() == 10

    # Biconnected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_firm'}))
    assert bdf.n_firms() == 10

def test_connectedness_4():
    # Test connected and leave-one-firm-out for collapsed long format.
    # There are 2 biconnected sets that are connected by 1 mover, so the largest connected set is all the observations. Unlike test 3, the largest biconnected component is group 1. This is because individual 2 goes from firm 2 to 3 to 6. We also now have individual 4 going from 4 to 5 to 6. However, removing firm 6 disconnects the 2 groups. Additionally, these new linkages to firm 6 mean firm 6 is a member of both groups.
    worker_data = []
    # Group 1 is firms 0 to 5
    # Worker 0
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 1, 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 2})
    # Worker 1
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 2, 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1, 't': 2})
    # Worker 2
    # Firm 2 -> 3 -> 6
    worker_data.append({'i': 2, 'j': 2, 'y': 3, 't': 1})
    worker_data.append({'i': 2, 'j': 3, 'y': 2.5, 't': 2})
    worker_data.append({'i': 2, 'j': 6, 'y': 1, 't': 3})
    # Worker 3
    # Firm 3 -> 4
    worker_data.append({'i': 3, 'j': 3, 'y': 1, 't': 1})
    worker_data.append({'i': 3, 'j': 4, 'y': 1, 't': 2})
    # Worker 4
    # Firm 4 -> 5 -> 6
    worker_data.append({'i': 4, 'j': 4, 'y': 1, 't': 1})
    worker_data.append({'i': 4, 'j': 5, 'y': 1.3, 't': 2})
    worker_data.append({'i': 4, 'j': 6, 'y': 1.15, 't': 3})
    # Worker 5
    # Firm 5 -> 0
    worker_data.append({'i': 5, 'j': 5, 'y': 1.1, 't': 1})
    worker_data.append({'i': 5, 'j': 0, 'y': 1, 't': 2})
    # Group 2 is firms 6 to 9
    # Group 2 is linked to Group 1 through firms 2, 3, and 6
    # Worker 6
    # Firm 6 -> 7
    worker_data.append({'i': 6, 'j': 6, 'y': 1, 't': 1})
    worker_data.append({'i': 6, 'j': 7, 'y': 2, 't': 2})
    # Worker 7
    # Firm 7 -> 8
    worker_data.append({'i': 7, 'j': 7, 'y': 1.5, 't': 1})
    worker_data.append({'i': 7, 'j': 8, 'y': 1.2, 't': 2})
    # Worker 8
    # Firm 8 -> 9
    worker_data.append({'i': 8, 'j': 8, 'y': 1.6, 't': 1})
    worker_data.append({'i': 8, 'j': 9, 'y': 2, 't': 2})
    # Worker 9
    # Firm 9 -> 6
    worker_data.append({'i': 9, 'j': 9, 'y': 1.8, 't': 1})
    worker_data.append({'i': 9, 'j': 6, 'y': 1.4, 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    # Connected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'connected'}))
    assert bdf.n_firms() == 10

    # Biconnected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_firm'}))
    assert bdf.n_firms() == 7

def test_connectedness_5():
    # Now looking at leave-one-observation-out
    # Test connected and leave-one-observation-out for collapsed long format.
    # There are 2 biconnected sets that are connected by 1 mover, so the largest connected set is all the observations. Unlike test 4, the largest biconnected component is all observations. This is because the largest leave-one-observation-out component does not drop the entire firm.
    worker_data = []
    # Group 1 is firms 0 to 5
    # Worker 0
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 1, 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 2})
    # Worker 1
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 2, 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1, 't': 2})
    # Worker 2
    # Firm 2 -> 3 -> 6
    worker_data.append({'i': 2, 'j': 2, 'y': 3, 't': 1})
    worker_data.append({'i': 2, 'j': 3, 'y': 2.5, 't': 2})
    worker_data.append({'i': 2, 'j': 6, 'y': 1, 't': 3})
    # Worker 3
    # Firm 3 -> 4
    worker_data.append({'i': 3, 'j': 3, 'y': 1, 't': 1})
    worker_data.append({'i': 3, 'j': 4, 'y': 1, 't': 2})
    # Worker 4
    # Firm 4 -> 5 -> 6
    worker_data.append({'i': 4, 'j': 4, 'y': 1, 't': 1})
    worker_data.append({'i': 4, 'j': 5, 'y': 1.3, 't': 2})
    worker_data.append({'i': 4, 'j': 6, 'y': 1.15, 't': 3})
    # Worker 5
    # Firm 5 -> 0
    worker_data.append({'i': 5, 'j': 5, 'y': 1.1, 't': 1})
    worker_data.append({'i': 5, 'j': 0, 'y': 1, 't': 2})
    # Group 2 is firms 6 to 9
    # Group 2 is linked to Group 1 through firms 2, 3, and 6
    # Worker 6
    # Firm 6 -> 7
    worker_data.append({'i': 6, 'j': 6, 'y': 1, 't': 1})
    worker_data.append({'i': 6, 'j': 7, 'y': 2, 't': 2})
    # Worker 7
    # Firm 7 -> 8
    worker_data.append({'i': 7, 'j': 7, 'y': 1.5, 't': 1})
    worker_data.append({'i': 7, 'j': 8, 'y': 1.2, 't': 2})
    # Worker 8
    # Firm 8 -> 9
    worker_data.append({'i': 8, 'j': 8, 'y': 1.6, 't': 1})
    worker_data.append({'i': 8, 'j': 9, 'y': 2, 't': 2})
    # Worker 9
    # Firm 9 -> 6
    worker_data.append({'i': 9, 'j': 9, 'y': 1.8, 't': 1})
    worker_data.append({'i': 9, 'j': 6, 'y': 1.4, 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    # Connected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'connected'}))
    assert bdf.n_firms() == 10

    # Biconnected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_observation'}))
    assert bdf.n_firms() == 10

def test_connectedness_6():
    # Test connected and leave-one-observation-out for collapsed long format.
    # There are 2 biconnected sets that are connected by 1 mover, so the largest connected set is all the observations. Unlike test 5, the largest biconnected component is group 1. This is because we removed the move for worker 4 to firm 6, so now there is only a single observation that moves between groups 1 and 2.
    worker_data = []
    # Group 1 is firms 0 to 5
    # Worker 0
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 1, 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 2})
    # Worker 1
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 2, 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1, 't': 2})
    # Worker 2
    # Firm 2 -> 3 -> 6
    worker_data.append({'i': 2, 'j': 2, 'y': 3, 't': 1})
    worker_data.append({'i': 2, 'j': 3, 'y': 2.5, 't': 2})
    worker_data.append({'i': 2, 'j': 6, 'y': 1, 't': 3})
    # Worker 3
    # Firm 3 -> 4
    worker_data.append({'i': 3, 'j': 3, 'y': 1, 't': 1})
    worker_data.append({'i': 3, 'j': 4, 'y': 1, 't': 2})
    # Worker 4
    # Firm 4 -> 5
    worker_data.append({'i': 4, 'j': 4, 'y': 1, 't': 1})
    worker_data.append({'i': 4, 'j': 5, 'y': 1.3, 't': 2})
    # Worker 5
    # Firm 5 -> 0
    worker_data.append({'i': 5, 'j': 5, 'y': 1.1, 't': 1})
    worker_data.append({'i': 5, 'j': 0, 'y': 1, 't': 2})
    # Group 2 is firms 6 to 9
    # Group 2 is linked to Group 1 through firms 2, 3, and 6
    # Worker 6
    # Firm 6 -> 7
    worker_data.append({'i': 6, 'j': 6, 'y': 1, 't': 1})
    worker_data.append({'i': 6, 'j': 7, 'y': 2, 't': 2})
    # Worker 7
    # Firm 7 -> 8
    worker_data.append({'i': 7, 'j': 7, 'y': 1.5, 't': 1})
    worker_data.append({'i': 7, 'j': 8, 'y': 1.2, 't': 2})
    # Worker 8
    # Firm 8 -> 9
    worker_data.append({'i': 8, 'j': 8, 'y': 1.6, 't': 1})
    worker_data.append({'i': 8, 'j': 9, 'y': 2, 't': 2})
    # Worker 9
    # Firm 9 -> 6
    worker_data.append({'i': 9, 'j': 9, 'y': 1.8, 't': 1})
    worker_data.append({'i': 9, 'j': 6, 'y': 1.4, 't': 2})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    # Connected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'connected'}))
    assert bdf.n_firms() == 10

    # Biconnected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_observation'}))
    assert bdf.n_firms() == 6
    assert bdf.n_workers() == 6

def test_connectedness_7():
    # Test connected and leave-one-observation-out for collapsed long format.
    # There are 2 biconnected sets that are connected by 1 mover, so the largest connected set is all the observations. Unlike test 6, the link from group 1 to 2 is a worker going from group 2 to group 1. The worker's observation in group 1 should remain, despite the fact that it is an articulation observation.
    worker_data = []
    # Group 1 is firms 0 to 5
    # Worker 0
    # Firm 0 -> 1
    worker_data.append({'i': 0, 'j': 0, 'y': 1, 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 2})
    # Worker 1
    # Firm 1 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 2, 't': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1, 't': 2})
    # Worker 2
    # Firm 2 -> 3
    worker_data.append({'i': 2, 'j': 2, 'y': 3, 't': 1})
    worker_data.append({'i': 2, 'j': 3, 'y': 2.5, 't': 2})
    # Worker 3
    # Firm 3 -> 4
    worker_data.append({'i': 3, 'j': 3, 'y': 1, 't': 1})
    worker_data.append({'i': 3, 'j': 4, 'y': 1, 't': 2})
    # Worker 4
    # Firm 4 -> 5
    worker_data.append({'i': 4, 'j': 4, 'y': 1, 't': 1})
    worker_data.append({'i': 4, 'j': 5, 'y': 1.3, 't': 2})
    # Worker 5
    # Firm 5 -> 0
    worker_data.append({'i': 5, 'j': 5, 'y': 1.1, 't': 1})
    worker_data.append({'i': 5, 'j': 0, 'y': 1, 't': 2})
    # Group 2 is firms 6 to 9
    # Group 2 is linked to Group 1 through firms 2, 3, and 6
    # Worker 6
    # Firm 6 -> 7
    worker_data.append({'i': 6, 'j': 6, 'y': 1, 't': 1})
    worker_data.append({'i': 6, 'j': 7, 'y': 2, 't': 2})
    # Worker 7
    # Firm 7 -> 8
    worker_data.append({'i': 7, 'j': 7, 'y': 1.5, 't': 1})
    worker_data.append({'i': 7, 'j': 8, 'y': 1.2, 't': 2})
    # Worker 8
    # Firm 8 -> 9
    worker_data.append({'i': 8, 'j': 8, 'y': 1.6, 't': 1})
    worker_data.append({'i': 8, 'j': 9, 'y': 2, 't': 2})
    # Worker 9
    # Firm 9 -> 6 -> 3
    worker_data.append({'i': 9, 'j': 9, 'y': 1.8, 't': 1})
    worker_data.append({'i': 9, 'j': 6, 'y': 1.4, 't': 2})
    worker_data.append({'i': 9, 'j': 3, 'y': 1.3, 't': 3})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    # Connected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'connected'}))
    assert bdf.n_firms() == 10

    # Biconnected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_observation'}))
    assert bdf.n_firms() == 6
    # Make sure articulation observations aren't dropped
    assert bdf.n_workers() == 7

def test_connectedness_8():
    '''
    This test uses simulated data that is very badly behaved. When computing the leave-one-observation-out connected set, if you drop firms with fewer than 2 moves then recollapse, this creates new firms with fewer than 2 moves. This test checks that the method is handling this correctly. In this data, if the cleaning process doesn't loop until convergence, the cleaned data will have firms with only 1 move.
    '''
    with open('tests/test_data/wid_drops.pkl', 'rb') as f:
        wid_drops = pickle.load(f)
    bad_df = bpd.BipartiteLongCollapsed(pd.read_feather('tests/test_data/bad_df.ftr')).drop_ids('i', wid_drops, copy=True)._reset_attributes(columns_contig=True, connected=True, no_na=False, no_duplicates=False, i_t_unique=False).clean(bpd.clean_params({'connectedness': 'leave_out_observation', 'force': True}))
    bad_df2 = bad_df.min_moves_frame(2)

    assert len(bad_df) == len(bad_df2) > 0

def test_connectedness_9():
    # Construct data that has different connected, leave-one-observation-out, leave-one-spell-out, leave-one-match-out, and leave-one-work-out connected components, and make sure each method is working properly.
    worker_data = []
    ## Group 1 is firms 0 to 2 ##
    # Worker 0
    # Firm 0 -> 1 -> 3 -> 3 -> 2
    worker_data.append({'i': 0, 'j': 0, 'y': 1, 't': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 2})
    worker_data.append({'i': 0, 'j': 3, 'y': 2.5, 't': 3})
    worker_data.append({'i': 0, 'j': 3, 'y': 1.5, 't': 4})
    worker_data.append({'i': 0, 'j': 2, 'y': 1, 't': 5})
    # Worker 1
    # Firm 1 -> 0 -> 2
    worker_data.append({'i': 1, 'j': 1, 'y': 1, 't': 1})
    worker_data.append({'i': 1, 'j': 0, 'y': 1.5, 't': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 2.5, 't': 3})
    ## Group 2 is firms 3 to 5 ##
    # Worker 2
    # Firm 3 -> 4 -> 5
    worker_data.append({'i': 2, 'j': 3, 'y': 2, 't': 1})
    worker_data.append({'i': 2, 'j': 4, 'y': 1, 't': 2})
    worker_data.append({'i': 2, 'j': 5, 'y': 2, 't': 3})
    # Worker 3
    # Firm 4 -> 7 -> 3 -> 7 -> 5
    worker_data.append({'i': 3, 'j': 4, 'y': 2.5, 't': 1})
    worker_data.append({'i': 3, 'j': 7, 'y': 1.75, 't': 2})
    worker_data.append({'i': 3, 'j': 3, 'y': 1, 't': 3})
    worker_data.append({'i': 3, 'j': 7, 'y': 1.75, 't': 4})
    worker_data.append({'i': 3, 'j': 5, 'y': 2, 't': 5})
    ## Group 3 is firms 6 to 8 ##
    # Worker 4
    # Firm 6 -> 7 -> 8
    worker_data.append({'i': 4, 'j': 6, 'y': 1, 't': 1})
    worker_data.append({'i': 4, 'j': 7, 'y': 1, 't': 2})
    worker_data.append({'i': 4, 'j': 8, 'y': 1.5, 't': 3})
    # Worker 5
    # Firm 7 -> 6 -> 8 -> 10
    worker_data.append({'i': 5, 'j': 7, 'y': 1.5, 't': 1})
    worker_data.append({'i': 5, 'j': 6, 'y': 2.25, 't': 2})
    worker_data.append({'i': 5, 'j': 8, 'y': 1.75, 't': 3})
    worker_data.append({'i': 5, 'j': 10, 'y': 2, 't': 4})
    ## Group 4 is firms 9 to 11 ##
    # Worker 6
    # Firm 9 -> 10 -> 11
    worker_data.append({'i': 6, 'j': 9, 'y': 1.25, 't': 1})
    worker_data.append({'i': 6, 'j': 10, 'y': 1, 't': 2})
    worker_data.append({'i': 6, 'j': 11, 'y': 1.5, 't': 3})
    # Worker 7
    # Firm 10 -> 9 -> 11
    worker_data.append({'i': 7, 'j': 10, 'y': 1.35, 't': 1})
    worker_data.append({'i': 7, 'j': 9, 'y': 0.75, 't': 2})
    worker_data.append({'i': 7, 'j': 11, 'y': 2.25, 't': 3})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    ## First, test on non-collapsed data ##
    # Connected
    bdf = bpd.BipartiteLong(df).clean(bpd.clean_params({'connectedness': 'connected'}))
    assert bdf.n_firms() == 12
    assert bdf.n_workers() == 8

    # Leave-out-observation
    bdf = bpd.BipartiteLong(df).clean(bpd.clean_params({'connectedness': 'leave_out_observation'}))
    assert bdf.n_firms() == 9
    assert bdf.n_workers() == 6

    # Leave-out-spell
    bdf = bpd.BipartiteLong(df).clean(bpd.clean_params({'connectedness': 'leave_out_spell'}))
    assert bdf.n_firms() == 6
    assert bdf.n_workers() == 5

    # Leave-out-match
    bdf = bpd.BipartiteLong(df).clean(bpd.clean_params({'connectedness': 'leave_out_match'}))
    assert bdf.n_firms() == 3
    assert bdf.n_workers() == 2

    # Leave-out-worker (note: 3 workers because leave-out-workers sorts components by number of firms + number of workers)
    bdf = bpd.BipartiteLong(df).clean(bpd.clean_params({'connectedness': 'leave_out_worker'}))
    assert bdf.n_firms() == 3
    assert bdf.n_workers() == 3

    ## Second, test on collapsed data ##
    # Connected
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'connected'}))
    assert bdf.n_firms() == 12
    assert bdf.n_workers() == 8

    # Leave-out-observation
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_observation'}))
    assert bdf.n_firms() == 6
    assert bdf.n_workers() == 5

    # Leave-out-spell
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_spell'}))
    assert bdf.n_firms() == 6
    assert bdf.n_workers() == 5

    # Leave-out-match
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_match'}))
    assert bdf.n_firms() == 3
    assert bdf.n_workers() == 2

    # Leave-out-worker (note: 3 workers because leave-out-workers sorts components by number of firms + number of workers)
    bdf = bpd.BipartiteLong(df).clean().collapse().clean(bpd.clean_params({'connectedness': 'leave_out_worker'}))
    assert bdf.n_firms() == 3
    assert bdf.n_workers() == 3

################################
##### Tests for Clustering #####
################################

def test_cluster_1():
    # Test cluster function is working correctly for long format.
    nk = 10
    sim_data = bpd.SimBipartite(bpd.sim_params({'nk': nk})).simulate(np.random.default_rng(2345))
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}))

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
    sim_data = bpd.SimBipartite(bpd.sim_params({'nk': nk})).simulate(np.random.default_rng(3456))
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()
    bdf = bdf.to_eventstudy()

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}))

            to_long = bdf.to_long()
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
    sim_data = bpd.SimBipartite(bpd.sim_params({'nk': nk})).simulate(np.random.default_rng(4567))
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()
    bdf = bdf.collapse()
    # Compute spells
    sim_data['i_l1'] = sim_data['i'].shift(1)
    sim_data['j_l1'] = sim_data['j'].shift(1)
    sim_data['new_spell'] = (sim_data['i'] != sim_data['i_l1']) | (sim_data['j'] != sim_data['j_l1'])
    sim_data['spell'] = sim_data['new_spell'].cumsum()
    sim_data_spell = sim_data.groupby('spell').first()

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}))
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
    sim_data = bpd.SimBipartite(bpd.sim_params({'nk': nk})).simulate(np.random.default_rng(5678))
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()
    bdf = bdf.collapse().to_eventstudy()
    # Compute spells
    sim_data['i_l1'] = sim_data['i'].shift(1)
    sim_data['j_l1'] = sim_data['j'].shift(1)
    sim_data['new_spell'] = (sim_data['i'] != sim_data['i_l1']) | (sim_data['j'] != sim_data['j_l1'])
    sim_data['spell'] = sim_data['new_spell'].cumsum()
    sim_data_spell = sim_data.groupby('spell').first()

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}))

            collapse = bdf.to_long()
            remaining_jids = collapse.dropna()['j'].unique()

            clusters_true = sim_data_spell[sim_data_spell['j'].isin(remaining_jids)]['psi'].astype('category').cat.codes.astype(int).to_numpy() # Skip firms that aren't clustered
            clusters_estimated = collapse[~collapse['g'].isna()]['g'].astype(int).to_numpy() # Skip firms that aren't clustered

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
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'g': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2, 'g': 2})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'g': 2})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'g': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 2., 't': 3, 'g': 1})
    worker_data.append({'i': 1, 'j': 5, 'y': 1., 't': 4, 'g': 3})
    # Worker 3
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1, 'g': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 2, 'g': 1})
    # Worker 4
    worker_data.append({'i': 4, 'j': 0, 'y': 1., 't': 1, 'g': 1})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(data=df)
    bdf = bdf.clean()

    measures = bpd.measures.Moments(measures='mean')
    grouping = bpd.grouping.Quantiles(n_quantiles=3)
    bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping}))

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

    ## Second, construct while adding column, but don't make it contiguous ##
    bdf = bpd.BipartiteLong(data=df).add_column('c').clean()

    assert 'c' in bdf.columns
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()
    assert bdf.n_unique_ids('c') != bdf['c'].max() + 1

    ## Third, construct while adding column, and make it contiguous ##
    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True).add_column('c', is_contiguous=True, dtype='contig').clean()

    assert 'c' in bdf.columns
    assert 'c' in bdf.id_reference_dict.keys()
    assert 'c' in bdf.original_ids().columns
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()
    assert bdf.n_unique_ids('c') == bdf['c'].max() + 1

    ## Fourth, construct while adding column, and make it contiguous but with an invalid collapse option ##
    try:
        bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True).add_column('c', is_contiguous=True, dtype='contig', how_collapse='mean').clean()
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

    ## Seventh, try with event study format, but don't make custom columns contiguous ##
    bdf = bpd.BipartiteEventStudy(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c').clean().to_eventstudy())).add_column('c').clean()

    assert 'c1' in bdf.columns and 'c2' in bdf.columns
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()
    assert bdf.n_unique_ids('c') != max(bdf['c1'].max(), bdf['c2'].max()) + 1

    ## Eighth, try with event study format, and make custom columns contiguous ##
    bdf = bpd.BipartiteEventStudy(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c').clean().to_eventstudy()), include_id_reference_dict=True).add_column('c', is_contiguous=True, dtype='contig').clean()

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

    ## Twelfth, go from long to event study with contiguous column that should drop ##
    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True).add_column('c', is_contiguous=True, dtype='contig', long_es_split=None).clean().to_eventstudy()

    assert ('c1' not in bdf.columns) and ('c2' not in bdf.columns)
    assert 'c' not in bdf.col_reference_dict.keys()
    assert 'c' not in bdf.id_reference_dict.keys()
    assert 'c' not in bdf.columns_contig.keys()
    assert 'c' not in bdf.col_dtype_dict.keys()
    assert 'c' not in bdf.col_collapse_dict.keys()
    assert 'c' not in bdf.col_long_es_dict.keys()

    ## Thirteenth, go from event study to long with contiguous column that should drop ##
    bdf = bpd.BipartiteEventStudy(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c').clean().to_eventstudy()), include_id_reference_dict=True).add_column('c', is_contiguous=True, dtype='contig', long_es_split=None).clean().to_long()

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
    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True).add_column('c', is_contiguous=True, dtype='contig', how_collapse=None).clean().collapse()

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
    bdf = bpd.BipartiteLongCollapsed(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c', how_collapse='mean').clean().collapse()), include_id_reference_dict=True).add_column('c', is_contiguous=True, dtype='contig', how_collapse=None).clean()

    assert 'c' in bdf.columns
    assert 'c' in bdf.col_reference_dict.keys()
    assert 'c' in bdf.id_reference_dict.keys()
    assert 'c' in bdf.columns_contig.keys()
    assert 'c' in bdf.col_dtype_dict.keys()
    assert 'c' in bdf.col_collapse_dict.keys()
    assert 'c' in bdf.col_long_es_dict.keys()

    ## Ninth, uncollapse by None ##
    bdf = bpd.BipartiteLongCollapsed(pd.DataFrame(bpd.BipartiteLong(data=df).add_column('c', how_collapse='mean').clean().collapse()), include_id_reference_dict=True).add_column('c', is_contiguous=True, dtype='contig', how_collapse=None).clean().uncollapse()

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

    ## First, contiguous to not contiguous ##
    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True).add_column('c', is_contiguous=True, dtype='contig').clean()
    init_properties = bdf.get_column_properties('c')

    assert (init_properties['dtype'] == 'contig') and init_properties['is_contiguous'] and (init_properties['how_collapse'] == 'first') and init_properties['long_es_split']
    assert ('c' in bdf.columns_contig.keys()) and ('c' in bdf.id_reference_dict.keys())

    bdf = bdf.set_column_properties('c', dtype='int', is_contiguous=False, how_collapse='mean', long_es_split=None)
    new_properties = bdf.get_column_properties('c')

    assert (new_properties['dtype'] == 'int') and (not new_properties['is_contiguous']) and (new_properties['how_collapse'] == 'mean') and (new_properties['long_es_split'] is None)
    assert ('c' not in bdf.columns_contig.keys()) and ('c' not in bdf.id_reference_dict.keys())

    ## Second, not contiguous to contiguous ##
    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True).add_column('c', is_contiguous=False).clean()
    init_properties = bdf.get_column_properties('c')

    assert (init_properties['dtype'] == 'any') and (not init_properties['is_contiguous']) and (init_properties['how_collapse'] == 'first') and init_properties['long_es_split']
    assert ('c' not in bdf.columns_contig.keys()) and ('c' not in bdf.id_reference_dict.keys())

    bdf = bdf.set_column_properties('c', is_contiguous=True, dtype='contig', how_collapse='last', long_es_split=None)
    new_properties = bdf.get_column_properties('c')

    assert (new_properties['dtype'] == 'contig') and new_properties['is_contiguous'] and (new_properties['how_collapse'] == 'last') and (new_properties['long_es_split'] is None)
    assert ('c' in bdf.columns_contig.keys()) and ('c' in bdf.id_reference_dict.keys())

    ## Third, not contiguous to contiguous, but invalid how_collapse ##
    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True).add_column('c', is_contiguous=False).clean()
    init_properties = bdf.get_column_properties('c')

    try:
        bdf = bdf.set_column_properties('c', is_contiguous=True, dtype='contig', how_collapse='mean', long_es_split=None)
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

    ## Contiguous column should alter id_reference_dict properly ##
    bdf = bpd.BipartiteLong(data=df, include_id_reference_dict=True).add_column('c', is_contiguous=True, dtype='contig').clean()

    assert 'c' in bdf.id_reference_dict.keys()

    bdf = bdf.rename({'c': 'cw'}, axis=1)

    assert 'cw' in bdf.id_reference_dict.keys()

    try:
        bdf = bdf.rename({'cw': 'cw1'}, axis=1)
        success = False
    except NotImplementedError:
        success = True

    assert success

########################################
##### Tests for BipartiteDataFrame #####
########################################

def test_dataframe_1():
    # Test BipartiteDataFrame constructor for different formats

    ## Long format ##
    a = bpd.SimBipartite().simulate()
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
    b = bpd.BipartiteDataFrame(**a, custom_contiguous_dict={'l': True}, custom_dtype_dict={'l': 'contig'}, custom_how_collapse_dict={'alpha': None, 'l': None}, custom_long_es_split_dict={'psi': False}).clean()

    assert 'l' in b.columns_contig.keys()
    assert b.col_dtype_dict['l'] == 'contig'
    assert b.col_collapse_dict['alpha'] is None
    assert b.col_long_es_dict['psi'] is False
