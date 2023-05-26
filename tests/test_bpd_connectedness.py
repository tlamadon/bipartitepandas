'''
Tests for bipartitepandas.
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pickle

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
    # Construct data that has different connected, strongly connected, leave-one-observation-out, leave-one-spell-out, leave-one-match-out, and leave-one-work-out connected components, and make sure each method is working properly.
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

    # Strongly-connected
    bdf = bpd.BipartiteLong(df).clean(bpd.clean_params({'connectedness': 'strongly_connected'}))
    assert bdf.n_firms() == 4
    assert bdf.n_workers() == 5

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

def test_connectedness_collapsed():
    # Test that collapsing data maintains connectedness properly
    quiet = bpd.clean_params({'verbose': False})

    # Construct data
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
    bdf = bpd.BipartiteDataFrame(i=df['i'], j=df['j'], y=df['y'], t=df['t'])

    ### Compute connectendess, then collapse ###
    ## Collapse at spell level ##
    # leave-out-observation
    loo_s = bdf.clean(bpd.clean_params({'connectedness': 'leave_out_observation', 'verbose': False})).collapse(level='spell')
    # leave-out-spell
    los_s = bdf.clean(bpd.clean_params({'connectedness': 'leave_out_spell', 'verbose': False})).collapse(level='spell')
    # leave-out-match
    lom_s = bdf.clean(bpd.clean_params({'connectedness': 'leave_out_match', 'verbose': False})).collapse(level='spell')
    ## Collapse at match level ##
    # leave-out-observation
    loo_m = bdf.clean(bpd.clean_params({'connectedness': 'leave_out_observation', 'verbose': False})).collapse(level='match')
    # leave-out-spell
    los_m = bdf.clean(bpd.clean_params({'connectedness': 'leave_out_spell', 'verbose': False})).collapse(level='match')
    # leave-out-match
    lom_m = bdf.clean(bpd.clean_params({'connectedness': 'leave_out_match', 'verbose': False})).collapse(level='match')

    ### Collapse (spell), then compute connectedness ###
    # leave-out-observation
    collapse_s_loo = bdf.clean(quiet).collapse(level='spell').clean(bpd.clean_params({'connectedness': 'leave_out_observation', 'verbose': False}))
    # leave-out-spell
    collapse_s_los = bdf.clean(quiet).collapse(level='spell').clean(bpd.clean_params({'connectedness': 'leave_out_spell', 'verbose': False}))
    # leave-out-match
    collapse_s_lom = bdf.clean(quiet).collapse(level='spell').clean(bpd.clean_params({'connectedness': 'leave_out_match', 'verbose': False}))

    ### Collapse (match), then compute connectedness ###
    # leave-out-observation
    collapse_m_loo = bdf.clean(quiet).collapse(level='match').clean(bpd.clean_params({'connectedness': 'leave_out_observation', 'verbose': False}))
    # leave-out-spell
    collapse_m_los = bdf.clean(quiet).collapse(level='match').clean(bpd.clean_params({'connectedness': 'leave_out_spell', 'verbose': False}))
    # leave-out-match
    collapse_m_lom = bdf.clean(quiet).collapse(level='match').clean(bpd.clean_params({'connectedness': 'leave_out_match', 'verbose': False}))

    ## Auto-collapse ##
    # leave-out-spell
    los_auto = bdf.clean(bpd.clean_params({'connectedness': 'leave_out_spell', 'collapse_at_connectedness_measure': True, 'verbose': False}))
    # leave-out-match
    lom_auto = bdf.clean(bpd.clean_params({'connectedness': 'leave_out_match', 'collapse_at_connectedness_measure': True, 'verbose': False}))

    ## Within group comparisons ##
    assert len(loo_s) > len(los_s) > len(lom_s)
    assert len(loo_m) > len(los_m) > len(lom_m)
    assert len(collapse_s_loo) == len(collapse_s_los) > len(collapse_s_lom)
    assert len(collapse_m_loo) == len(collapse_m_los) == len(collapse_m_lom)

    ## Between group comparisons ##
    # Collapsing at match should make it shorter than collapsing at spell
    assert len(loo_m) < len(loo_s)
    assert len(los_m) < len(los_s)
    # Collapsing at spell level then computing leave-out-observation should be equivalent to computing leave-out-spell then collapsing at spell level
    assert len(los_s) == len(collapse_s_loo)
    # Collapsing at match level then computing leave-out-observation should be equivalent to either computing leave-out-match and collapsing, or collapsing at spell level then computing leave-out-match
    assert len(lom_s) == len(lom_m) == len(collapse_s_lom) == len(collapse_m_loo)
    # Auto should be equivalent to cleaning, collapsing, then computing leave-out-observation
    assert len(los_auto) == len(collapse_s_loo)
    assert len(lom_auto) == len(collapse_m_loo)

def test_connectedness_strongly_loo():
    # Test that strongly-leave-out-x connectedness works
    sim_params = bpd.sim_params({'n_workers': 5000, 'firm_size': 10, 'p_move': 0.05})
    sim_data = bpd.SimBipartite(sim_params).simulate(rng=np.random.default_rng(1234))

    for measure in ['observation', 'spell', 'match', 'worker']:
        clean_params_loo = bpd.clean_params(
            {
                'connectedness': f'leave_out_{measure}',
                'collapse_at_connectedness_measure': True,
                'drop_single_stayers': True,
                'drop_returns': 'returners',
                'copy': True
            }
        )
        clean_params_strong = bpd.clean_params(
            {
                'connectedness': 'strongly_connected',
                'collapse_at_connectedness_measure': True,
                'drop_single_stayers': True,
                'drop_returns': 'returners',
                'copy': True
            }
        )

        #########################################################
        ## Strongly connected, but not leave-one-out connected ##
        #########################################################
        clean_params2 = bpd.clean_params(
            {
                'connectedness': 'strongly_connected',
                'collapse_at_connectedness_measure': True,
                'drop_single_stayers': True,
                'drop_returns': 'returners',
                'copy': False
            }
        )

        # Convert into BipartitePandas DataFrame
        bdf2 = bpd.BipartiteDataFrame(sim_data)
        # Clean
        bdf2 = bdf2.clean(clean_params2)

        # Check that it's not leave-one-out connected
        bdf2_loo = bdf2.clean(clean_params_loo)
        assert len(bdf2) != len(bdf2_loo)

        #########################################################
        ## Leave-one-out connected, but not strongly connected ##
        #########################################################
        clean_params3 = bpd.clean_params(
            {
                'connectedness': f'leave_out_{measure}',
                'collapse_at_connectedness_measure': True,
                'drop_single_stayers': True,
                'drop_returns': 'returners',
                'copy': False
            }
        )

        # Convert into BipartitePandas DataFrame
        bdf3 = bpd.BipartiteDataFrame(sim_data)
        # Clean
        bdf3 = bdf3.clean(clean_params3)

        # Check that it's not strongly connected
        bdf3_strong = bdf3.clean(clean_params_strong)
        assert len(bdf3) != len(bdf3_strong)

        ######################################
        ## Strongly leave-one-out connected ##
        ######################################
        clean_params4 = bpd.clean_params(
            {
                'connectedness': f'strongly_leave_out_{measure}',
                'collapse_at_connectedness_measure': True,
                'drop_single_stayers': True,
                'drop_returns': 'returners',
                'copy': False
            }
        )

        # Convert into BipartitePandas DataFrame
        bdf4 = bpd.BipartiteDataFrame(sim_data)
        # Clean
        bdf4 = bdf4.clean(clean_params4)

        # Check that it's leave-one-out connected
        bdf4_loo = bdf4.clean(clean_params_loo)
        assert len(bdf4) == len(bdf4_loo)

        # Check that it's strongly connected
        bdf4_strong = bdf4.clean(clean_params_strong)
        assert len(bdf4) == len(bdf4_strong)
