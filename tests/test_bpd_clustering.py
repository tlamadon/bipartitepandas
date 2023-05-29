'''
Tests for bipartitepandas.
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd

################################
##### Tests for Clustering #####
################################

def test_cluster_1():
    # Test cluster function is working correctly for long format.
    rng = np.random.default_rng(2345)
    nk = 10
    sim_data = bpd.SimBipartite(bpd.sim_params({'nk': nk})).simulate(rng)
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()

    for measure in ['quantile_all', 'quantile_firm']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}), rng)

            clusters_true = sim_data[~bdf['g'].isna()]['psi'].astype('category', copy=False).cat.codes.astype(int, copy=False).to_numpy() # Skip firms that aren't clustered
            clusters_estimated = bdf[~bdf['g'].isna()]['g'].astype(int, copy=False).to_numpy() # Skip firms that aren't clustered

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
                bound = 1100 # 2.2% error
            elif measure == 'quantile_firm':
                bound = 1100 # 2.2% error
            if stayers_movers == 'stayers':
                bound = 35000 # 70% error

            assert wrong_cluster < bound, 'error is {} for {}'.format(wrong_cluster, measure)

def test_cluster_2():
    # Test cluster function is working correctly for event study format.
    rng = np.random.default_rng(3456)
    nk = 10
    sim_data = bpd.SimBipartite(bpd.sim_params({'nk': nk})).simulate(rng)
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()
    bdf = bdf.to_eventstudy()

    for measure in ['quantile_all', 'quantile_firm']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}), rng)

            to_long = bdf.to_long()
            clusters_true = sim_data[~to_long['g'].isna()]['psi'].astype('category', copy=False).cat.codes.astype(int, copy=False).to_numpy() # Skip firms that aren't clustered
            clusters_estimated = to_long[~to_long['g'].isna()]['g'].astype(int, copy=False).to_numpy() # Skip firms that aren't clustered

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
                bound = 1300 # 2.6% error
            elif measure == 'quantile_firm':
                bound = 2600 # 5.2% error
            if stayers_movers == 'stayers':
                bound = 35000 # 70% error

            assert wrong_cluster < bound, 'error is {} for {}'.format(wrong_cluster, measure)

def test_cluster_3():
    # Test cluster function is working correctly for collapsed long format.
    rng = np.random.default_rng(4567)
    nk = 10
    sim_data = bpd.SimBipartite(bpd.sim_params({'nk': nk})).simulate(rng)
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()
    bdf = bdf.collapse()
    # Compute spells
    sim_data['i_l1'] = sim_data['i'].shift(1)
    sim_data['j_l1'] = sim_data['j'].shift(1)
    sim_data['new_spell'] = (sim_data['i'] != sim_data['i_l1']) | (sim_data['j'] != sim_data['j_l1'])
    sim_data['spell'] = sim_data['new_spell'].cumsum()
    sim_data_spell = sim_data.groupby('spell').first()

    for measure in ['quantile_all', 'quantile_firm']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}), rng)
            remaining_jids = bdf.dropna()['j'].unique()

            clusters_true = sim_data_spell[sim_data_spell['j'].isin(remaining_jids)]['psi'].astype('category', copy=False).cat.codes.astype(int, copy=False).to_numpy() # Skip firms that aren't clustered
            clusters_estimated = bdf[~bdf['g'].isna()]['g'].astype(int, copy=False).to_numpy() # Skip firms that aren't clustered

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
                bound = 800 # 1.6% error
            elif measure == 'quantile_firm':
                bound = 1900 # 3.8% error
            if stayers_movers == 'stayers':
                bound = 35000 # 70% error

            assert wrong_cluster < bound, 'error is {} for {}'.format(wrong_cluster, measure)

def test_cluster_4():
    # Test cluster function is working correctly for collapsed event study format.
    rng = np.random.default_rng(5678)
    nk = 10
    sim_data = bpd.SimBipartite(bpd.sim_params({'nk': nk})).simulate(rng)
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()
    bdf = bdf.collapse().to_eventstudy()
    # Compute spells
    sim_data['i_l1'] = sim_data['i'].shift(1)
    sim_data['j_l1'] = sim_data['j'].shift(1)
    sim_data['new_spell'] = (sim_data['i'] != sim_data['i_l1']) | (sim_data['j'] != sim_data['j_l1'])
    sim_data['spell'] = sim_data['new_spell'].cumsum()
    sim_data_spell = sim_data.groupby('spell').first()

    for measure in ['quantile_all', 'quantile_firm']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}), rng)

            collapse = bdf.to_long()
            remaining_jids = collapse.dropna()['j'].unique()

            clusters_true = sim_data_spell[sim_data_spell['j'].isin(remaining_jids)]['psi'].astype('category', copy=False).cat.codes.astype(int, copy=False).to_numpy() # Skip firms that aren't clustered
            clusters_estimated = collapse[~collapse['g'].isna()]['g'].astype(int, copy=False).to_numpy() # Skip firms that aren't clustered

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
                bound = 900 # 1.8% error
            elif measure == 'quantile_firm':
                bound = 5000 # 10% error
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
    bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping}), np.random.default_rng(6789))

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
