'''
Tests for bipartitepandas.
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pickle

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

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}), rng)

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
    rng = np.random.default_rng(3456)
    nk = 10
    sim_data = bpd.SimBipartite(bpd.sim_params({'nk': nk})).simulate(rng)
    bdf = bpd.BipartiteLong(sim_data[['i', 'j', 'y', 't']])
    bdf = bdf.clean()
    bdf = bdf.to_eventstudy()

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}), rng)

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

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}), rng)
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

    for measure in ['quantile_all', 'quantile_firm_small', 'quantile_firm_large']:
        for stayers_movers in [None, 'stayers', 'movers']:
            measures = bpd.measures.CDFs(measure=measure)
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            bdf = bdf.cluster(bpd.cluster_params({'measures': measures, 'grouping': grouping, 'stayers_movers': stayers_movers}), rng)

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
    print('a.columns:', a.columns)
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
