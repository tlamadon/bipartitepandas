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
#     bdf = bpd.BipartiteLong(sim_data).clean().cluster(grouping=bpd.grouping.KMeans(n_clusters=2))
#     bdf.plot_extended_eventstudy()

#     assert True # Just making sure it doesn't crash
