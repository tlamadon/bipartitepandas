'''
Functions for computing cluster measures
'''
import numpy as np
from bipartitepandas import to_list, aggregate_transform
from statsmodels.stats.weightstats import DescrStatsW

def cdfs(cdf_resolution=10, measure='quantile_all'):
    '''
    Generate cdfs of compensation for firms. Used for clustering.

    Arguments:
        cdf_resolution (int): how many values to use to approximate the cdfs
        measure (str): how to compute the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary

    Returns:
        compute_measures_cdfs (function): subfunction
    '''
    # Workaround for multiprocessing
    # Source: https://stackoverflow.com/a/61879723
    global compute_measures_cdfs

    def compute_measures_cdfs(data, jids):
        '''
        Arguments:
            data (Pandas DataFrame): data to use
            jids (list): sorted list of firm ids in data (since data could be a subset of self, this is not necessarily all firms in self)
        Returns:
            cdfs (NumPy Array): NumPy array of firm cdfs
        '''
        # Initialize cdf array
        n_firms = len(jids) # Can't use self.n_firms() since data could be a subset of self
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

        # Group by income cdfs
        if measure == 'quantile_all':
            # Get quantiles from all data
            quantile_groups = DescrStatsW(data['y'], weights=data['row_weights']).quantile(quantiles, return_pandas=False)

            # Generate firm-level cdfs
            data.sort_values('j', inplace=True) # Required for aggregate_transform
            for i, quant in enumerate(quantile_groups):
                data['quant'] = (data['y'] <= quant).astype(int)
                cdfs_col = aggregate_transform(data, col_groupby='j', col_grouped='quant', func='sum', weights='row_weights', merge=False) # aggregate(data['fid'], firm_quant, func='sum', fill_value=- 1)
                cdfs[:, i] = cdfs_col[cdfs_col >= 0]
            data.drop('quant', axis=1, inplace=True)
            del cdfs_col

            # Normalize by firm size (convert to cdf)
            jsize = data.groupby('j')['row_weights'].sum().to_numpy()
            cdfs = (cdfs.T / jsize.T).T

        elif measure in ['quantile_firm_small', 'quantile_firm_large']:
            # Sort data by compensation (do this once now, so that don't need to do it again later) (also note it is faster to sort then manually compute quantiles than to use built-in quantile functions)
            data = data.sort_values(['y'])

            if measure == 'quantile_firm_small':
                # Convert pandas dataframe into a dictionary to access data faster
                # Source for idea: https://stackoverflow.com/questions/57208997/looking-for-the-fastest-way-to-slice-a-row-in-a-huge-pandas-dataframe
                # Source for how to actually format data correctly: https://stackoverflow.com/questions/56064677/pandas-series-to-dict-with-repeated-indices-make-dict-with-list-values
                # data_dict = data['y'].groupby(level=0).agg(list).to_dict()
                data_dict = data.groupby('j')['y'].agg(list).to_dict()
                weights_dict = data.groupby('j')['row_weights'].agg(list).to_dict()
                # data.sort_values(['j', 'y'], inplace=True) # Required for aggregate_transform
                # data_dict = pd.Series(aggregate_transform(data, col_groupby='j', col_grouped='y', func='array', merge=False), index=np.unique(data['j'])).to_dict()
                # with warnings.catch_warnings():
                #     warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
                #     data_dict = pd.Series(aggregate(data['j'], data['y'], func='array', fill_value=[]), index=np.unique(data['j'])).to_dict()

            # Generate the cdfs
            for i, jid in enumerate(jids):
                # Get the firm-level compensation data (don't need to sort because already sorted)
                if measure == 'quantile_firm_small':
                    y = np.array(data_dict[jid])
                    w = np.array(weights_dict[jid])
                elif measure == 'quantile_firm_large':
                    y = data.loc[data['j'] == jid, 'y'].to_numpy()
                    w = data.loc[data['j'] == jid, 'row_weights'].to_numpy()
                cum_w = w.cumsum() # Cumulative weight
                weighted_n = w.sum() # Weighted number of observations
                # Generate the firm-level cdf
                # Note: update numpy array element by element
                # Source: https://stackoverflow.com/questions/30012362/faster-way-to-convert-list-of-objects-to-numpy-array/30012403
                for j, quantile in enumerate(quantiles):
                    # index = max(len(y) * (j + 1) // cdf_resolution - 1, 0) # Don't want negative index
                    index = 0 # Income index at particular quantile
                    for cum_w_val in cum_w[1:]: # Skip first weight because it is always true
                        if cum_w_val / weighted_n <= quantile:
                            index += 1
                        else:
                            break
                    # Update cdfs with the firm-level cdf
                    cdfs[i, j] = y[index]

        return cdfs
    return compute_measures_cdfs

def moments(measures='mean'):
    '''
    Generate compensation moments for firms. Used for clustering.

    Arguments:
        measures (str or list of str): how to compute the measures ('mean' to compute average income within each firm; 'var' to compute variance of income within each firm; 'max' to compute max income within each firm; 'min' to compute min income within each firm)

    Returns:
        compute_measures_moments (function): subfunction
    '''
    # Workaround for multiprocessing
    # Source: https://stackoverflow.com/a/61879723
    global compute_measures_moments

    def compute_measures_moments(data, jids):
        '''
        Arguments:
            data (Pandas DataFrame): data to use
            jids (list): sorted list of firm ids in data (since data could be a subset of full dataset, this is not necessarily all firms in self)

        Returns:
            moments (NumPy Array): NumPy array of firm moments
        '''
        n_firms = len(jids) # Can't use data.n_firms() since data could be a subset of self
        n_measures = len(to_list(measures))
        moments = np.zeros([n_firms, n_measures])

        data.sort_values('j', inplace=True) # Required for aggregate_transform

        for j, measure in enumerate(to_list(measures)):
            if measure == 'mean':
                # Group by mean income
                data['one'] = 1
                moments[:, j] = aggregate_transform(data, 'j', 'y', 'sum', weights='row_weights', merge=False) / aggregate_transform(data, 'j', 'one', 'sum', weights='row_weights', merge=False)
            elif measure == 'var':
                # Group by variance of income
                moments[:, j] = aggregate_transform(data, 'j', 'y', 'var', weights='row_weights', merge=False)
            elif measure == 'max':
                moments[:, j] = data.groupby('j')['y'].max().to_numpy()
            elif measure == 'min':
                moments[:, j] = data.groupby('j')['y'].min().to_numpy()

        return moments
    return compute_measures_moments
