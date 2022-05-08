'''
Classes for computing cluster measures. Note: use classes rather than nested functions because nested functions cannot be pickled (source: https://stackoverflow.com/a/12022055/17333120).
'''
import numpy as np
from bipartitepandas.util import to_list, aggregate_transform
from statsmodels.stats.weightstats import DescrStatsW

class CDFs:
    '''
    Generate cdfs of compensation for firms. Used for clustering.

    Arguments:
        cdf_resolution (int): how many values to use to approximate the cdfs
        measure (str): how to compute the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary
    '''

    def __init__(self, cdf_resolution=10, measure='quantile_all'):
        self.cdf_resolution = cdf_resolution
        self.measure = measure

    def _compute_measure(self, frame, jids):
        '''
        Arguments:
            frame (Pandas DataFrame): data to use
            jids (list): sorted list of firm ids in frame (since frame could be a subset of self, this is not necessarily all firms in self)
        Returns:
            (NumPy Array): NumPy array of firm cdfs
        '''
        cdf_resolution = self.cdf_resolution
        measure = self.measure

        ## Initialize cdf array
        # Can't use self.n_firms() since data could be a subset of self
        n_firms = len(jids)
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

        # Group by income cdfs
        if measure == 'quantile_all':
            # Get quantiles from all data
            quantile_groups = DescrStatsW(frame.loc[:, 'y'].to_numpy(), weights=frame.loc[:, 'row_weights'].to_numpy()).quantile(quantiles, return_pandas=False)

            ## Generate firm-level cdfs
            # Required for aggregate_transform
            # NOTE: don't sort in-place, otherwise modifies external data
            frame = frame.sort_values('j', inplace=False)
            for i, quant in enumerate(quantile_groups):
                frame.loc[:, 'quant'] = (frame.loc[:, 'y'].to_numpy() <= quant).astype(int, copy=False)
                cdfs_col = aggregate_transform(frame, col_groupby='j', col_grouped='quant', func='sum', weights='row_weights', merge=False) # aggregate(frame['fid'], firm_quant, func='sum', fill_value=-1)
                cdfs[:, i] = cdfs_col[cdfs_col >= 0]
            frame.drop('quant', axis=1, inplace=True)
            del cdfs_col

            # Normalize by firm size (convert to cdf)
            jsize = frame.groupby('j', sort=False)['row_weights'].sum().to_numpy()
            cdfs = (cdfs.T / jsize.T).T

        elif measure in ['quantile_firm_small', 'quantile_firm_large']:
            # Sort frame by compensation (do this once now, so that don't need to do it again later) (also note it is faster to sort then manually compute quantiles than to use built-in quantile functions)
            # NOTE: don't sort in-place, otherwise modifies external data
            frame = frame.sort_values('y', inplace=False)

            if measure == 'quantile_firm_small':
                # Convert pandas dataframe into a dictionary to access data faster
                # Source for idea: https://stackoverflow.com/questions/57208997/looking-for-the-fastest-way-to-slice-a-row-in-a-huge-pandas-dataframe
                # Source for how to actually format data correctly: https://stackoverflow.com/questions/56064677/pandas-series-to-dict-with-repeated-indices-make-dict-with-list-values
                # frame_dict = frame['y'].groupby(level=0).agg(list).to_dict()
                frame_groupby_j = frame.groupby('j')
                frame_dict = frame_groupby_j['y'].agg(list).to_dict()
                weights_dict = frame_groupby_j['row_weights'].agg(list).to_dict()
                del frame_groupby_j
                # frame.sort_values(['j', 'y'], inplace=True) # Required for aggregate_transform
                # frame_dict = pd.Series(aggregate_transform(frame, col_groupby='j', col_grouped='y', func='array', merge=False), index=np.unique(frame['j'])).to_dict()
                # with warnings.catch_warnings():
                #     warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
                #     frame_dict = pd.Series(aggregate(frame['j'], frame['y'], func='array', fill_value=[]), index=np.unique(frame['j'])).to_dict()

            # Generate the cdfs
            for i, jid in enumerate(jids):
                # Get the firm-level compensation data (don't need to sort because already sorted)
                if measure == 'quantile_firm_small':
                    y = np.array(frame_dict[jid])
                    w = np.array(weights_dict[jid])
                elif measure == 'quantile_firm_large':
                    y = frame.loc[frame.loc[:, 'j'].to_numpy() == jid, 'y'].to_numpy()
                    w = frame.loc[frame.loc[:, 'j'].to_numpy() == jid, 'row_weights'].to_numpy()
                # Cumulative weight
                cum_w = w.cumsum()
                # Weighted number of observations
                weighted_n = w.sum()
                # Generate the firm-level cdf
                # Note: update numpy array element by element
                # Source: https://stackoverflow.com/questions/30012362/faster-way-to-convert-list-of-objects-to-numpy-array/30012403
                for j, quantile in enumerate(quantiles):
                    # index = max(len(y) * (j + 1) // cdf_resolution - 1, 0) # Don't want negative index
                    # Income index at particular quantile
                    index = 0
                    for cum_w_val in cum_w[1:]:
                        # Skip first weight because it is always true
                        if cum_w_val / weighted_n <= quantile:
                            index += 1
                        else:
                            break
                    # Update cdfs with the firm-level cdf
                    cdfs[i, j] = y[index]

        return cdfs

class Moments:
    '''
    Generate compensation moments for firms. Used for clustering.

    Arguments:
        measures (str or list of str): how to compute the measures ('mean' to compute average income within each firm; 'var' to compute variance of income within each firm; 'max' to compute max income within each firm; 'min' to compute min income within each firm)
    '''

    def __init__(self, measures='mean'):
        self.measures = measures

    def _compute_measure(self, frame, jids):
        '''
        Arguments:
            frame (Pandas DataFrame): data to use
            jids (list): sorted list of firm ids in frame (since frame could be a subset of full dataset, this is not necessarily all firms in self)

        Returns:
            (NumPy Array): NumPy array of firm moments
        '''
        measures = self.measures

        # Can't use data.n_firms() since data could be a subset of self
        n_firms = len(jids)
        n_measures = len(to_list(measures))
        moments = np.zeros([n_firms, n_measures])

        # Required for aggregate_transform
        # NOTE: don't sort in-place, otherwise modifies external data
        frame = frame.sort_values('j', inplace=False)

        for j, measure in enumerate(to_list(measures)):
            if measure == 'mean':
                # Group by mean income
                frame['one'] = 1
                moments[:, j] = aggregate_transform(frame, 'j', 'y', 'sum', weights='row_weights', merge=False) / aggregate_transform(frame, 'j', 'one', 'sum', weights='row_weights', merge=False)
            elif measure == 'var':
                # Group by variance of income
                moments[:, j] = aggregate_transform(frame, 'j', 'y', 'var', weights='row_weights', merge=False)
            elif measure == 'max':
                moments[:, j] = frame.groupby('j', sort=False)['y'].max().to_numpy()
            elif measure == 'min':
                moments[:, j] = frame.groupby('j', sort=False)['y'].min().to_numpy()

        return moments
