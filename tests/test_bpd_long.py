'''
Tests for bipartitepandas.
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pickle

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
#     bdf = bpd.BipartiteDataFrame(sim_data).clean().cluster(bpd.cluster_params({'grouping': bpd.grouping.KMeans(n_clusters=2)}))
#     bdf.plot_extended_eventstudy()

#     # Just making sure it doesn't crash
#     assert True

def test_weighted_collapse_1():
    # Test that collapsing weights properly at the spell level.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'w': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2, 'w': 2})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 3, 'w': 3})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4, 'w': 1})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'w': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'w': 1.8})
    worker_data.append({'i': 1, 'j': 2, 'y': 2., 't': 3, 'w': 2.5})
    worker_data.append({'i': 1, 'j': 5, 'y': 1., 't': 3, 'w': 1})
    # Worker 2
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1, 'w': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1.5, 't': 2, 'w': 2})
    worker_data.append({'i': 3, 'j': 3, 'y': 1.5, 't': 3, 'w': 1})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(df)
    bdf = bdf.clean().collapse(level='spell')

    # Worker 0
    assert bdf.iloc[0]['i'] == 0
    assert bdf.iloc[0]['j'] == 0
    assert bdf.iloc[0]['y'] == 2
    assert bdf.iloc[0]['t1'] == 1
    assert bdf.iloc[0]['t2'] == 1
    assert bdf.iloc[0]['w'] == 1

    assert bdf.iloc[1]['i'] == 0
    assert bdf.iloc[1]['j'] == 1
    assert bdf.iloc[1]['y'] == (2 * 1 + 3 * 1.5) / (2 + 3)
    assert bdf.iloc[1]['t1'] == 2
    assert bdf.iloc[1]['t2'] == 3
    assert bdf.iloc[1]['w'] == 5

    assert bdf.iloc[2]['i'] == 0
    assert bdf.iloc[2]['j'] == 0
    assert bdf.iloc[2]['y'] == 1
    assert bdf.iloc[2]['t1'] == 4
    assert bdf.iloc[2]['t2'] == 4
    assert bdf.iloc[2]['w'] == 1

    # Worker 1
    assert bdf.iloc[3]['i'] == 1
    assert bdf.iloc[3]['j'] == 1
    assert bdf.iloc[3]['y'] == 1
    assert bdf.iloc[3]['t1'] == 1
    assert bdf.iloc[3]['t2'] == 1
    assert bdf.iloc[3]['w'] == 1

    assert bdf.iloc[4]['i'] == 1
    assert bdf.iloc[4]['j'] == 2
    assert bdf.iloc[4]['y'] == (1.8 * 1 + 2.5 * 2) / (1.8 + 2.5)
    assert bdf.iloc[4]['t1'] == 2
    assert bdf.iloc[4]['t2'] == 3
    assert bdf.iloc[4]['w'] == 4.3

    # Worker 2
    assert bdf.iloc[5]['i'] == 2
    assert bdf.iloc[5]['j'] == 2
    assert bdf.iloc[5]['y'] == (1 * 1 + 2 * 1.5) / (1 + 2)
    assert bdf.iloc[5]['t1'] == 1
    assert bdf.iloc[5]['t2'] == 2
    assert bdf.iloc[5]['w'] == 3

    assert bdf.iloc[6]['i'] == 2
    assert bdf.iloc[6]['j'] == 3
    assert bdf.iloc[6]['y'] == 1.5
    assert bdf.iloc[6]['t1'] == 3
    assert bdf.iloc[6]['t2'] == 3
    assert bdf.iloc[6]['w'] == 1

def test_weighted_collapse_2():
    # Test that collapsing weights properly at the match level.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'w': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2, 'w': 2})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 3, 'w': 3})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4, 'w': 1})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'w': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'w': 1.8})
    worker_data.append({'i': 1, 'j': 2, 'y': 2., 't': 3, 'w': 2.5})
    worker_data.append({'i': 1, 'j': 5, 'y': 1., 't': 3, 'w': 1})
    # Worker 2
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1, 'w': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1.5, 't': 2, 'w': 2})
    worker_data.append({'i': 3, 'j': 3, 'y': 1.5, 't': 3, 'w': 1})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(df)
    bdf = bdf.clean().collapse(level='match')

    # Worker 0
    assert bdf.iloc[0]['i'] == 0
    assert bdf.iloc[0]['j'] == 0
    assert bdf.iloc[0]['y'] == 1.5
    assert bdf.iloc[0]['t1'] == 1
    assert bdf.iloc[0]['t2'] == 4
    assert bdf.iloc[0]['w'] == 2

    assert bdf.iloc[1]['i'] == 0
    assert bdf.iloc[1]['j'] == 1
    assert bdf.iloc[1]['y'] == (2 * 1 + 3 * 1.5) / (2 + 3)
    assert bdf.iloc[1]['t1'] == 2
    assert bdf.iloc[1]['t2'] == 3
    assert bdf.iloc[1]['w'] == 5

    # Worker 1
    assert bdf.iloc[2]['i'] == 1
    assert bdf.iloc[2]['j'] == 1
    assert bdf.iloc[2]['y'] == 1
    assert bdf.iloc[2]['t1'] == 1
    assert bdf.iloc[2]['t2'] == 1
    assert bdf.iloc[2]['w'] == 1

    assert bdf.iloc[3]['i'] == 1
    assert bdf.iloc[3]['j'] == 2
    assert bdf.iloc[3]['y'] == (1.8 * 1 + 2.5 * 2) / (1.8 + 2.5)
    assert bdf.iloc[3]['t1'] == 2
    assert bdf.iloc[3]['t2'] == 3
    assert bdf.iloc[3]['w'] == 4.3

    # Worker 2
    assert bdf.iloc[4]['i'] == 2
    assert bdf.iloc[4]['j'] == 2
    assert bdf.iloc[4]['y'] == (1 * 1 + 2 * 1.5) / (1 + 2)
    assert bdf.iloc[4]['t1'] == 1
    assert bdf.iloc[4]['t2'] == 2
    assert bdf.iloc[4]['w'] == 3

    assert bdf.iloc[5]['i'] == 2
    assert bdf.iloc[5]['j'] == 3
    assert bdf.iloc[5]['y'] == 1.5
    assert bdf.iloc[5]['t1'] == 3
    assert bdf.iloc[5]['t2'] == 3
    assert bdf.iloc[5]['w'] == 1

def test_weighted_collapse_2():
    # Test that collapsing computed weighted variance properly.
    worker_data = []
    # Worker 0
    worker_data.append({'i': 0, 'j': 0, 'y': 2., 't': 1, 'w': 1})
    worker_data.append({'i': 0, 'j': 1, 'y': 1., 't': 2, 'w': 2})
    worker_data.append({'i': 0, 'j': 1, 'y': 1.5, 't': 3, 'w': 3})
    worker_data.append({'i': 0, 'j': 0, 'y': 1., 't': 4, 'w': 1})
    # Worker 1
    worker_data.append({'i': 1, 'j': 1, 'y': 1., 't': 1, 'w': 1})
    worker_data.append({'i': 1, 'j': 2, 'y': 1., 't': 2, 'w': 1.8})
    worker_data.append({'i': 1, 'j': 2, 'y': 2., 't': 3, 'w': 2.5})
    worker_data.append({'i': 1, 'j': 5, 'y': 1., 't': 3, 'w': 1})
    # Worker 2
    worker_data.append({'i': 3, 'j': 2, 'y': 1., 't': 1, 'w': 1})
    worker_data.append({'i': 3, 'j': 2, 'y': 1.5, 't': 2, 'w': 2})
    worker_data.append({'i': 3, 'j': 3, 'y': 1.5, 't': 3, 'w': 1})

    df = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])

    bdf = bpd.BipartiteLong(df)
    bdf = bdf.clean()
    bdf.col_collapse_dict['y'] = 'var'
    bdf = bdf.collapse(level='spell')

    # Worker 0
    assert bdf.iloc[0]['y'] == 0
    y_mean = (2 * 1 + 3 * 1.5) / (2 + 3)
    assert bdf.iloc[1]['y'] == (2 * (1 - y_mean) ** 2 + 3 * (1.5 - y_mean) ** 2) / (2 + 3)
    assert bdf.iloc[2]['y'] == 0

    # Worker 1
    assert bdf.iloc[3]['y'] == 0
    y_mean = (1.8 * 1 + 2.5 * 2) / (1.8 + 2.5)
    assert bdf.iloc[4]['y'] == (1.8 * (1 - y_mean) ** 2 + 2.5 * (2 - y_mean) ** 2) / (1.8 + 2.5)

    # Worker 2
    y_mean = (1 * 1 + 2 * 1.5) / (1 + 2)
    assert bdf.iloc[5]['y'] == (1 * (1 - y_mean) ** 2 + 2 * (1.5 - y_mean) ** 2) / (1 + 2)
    assert bdf.iloc[6]['y'] == 0
