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
        measure (str): how to compute the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm' to get quantiles at the firm-level and have values be compensations)
        outcome_col (str): outcome_col column to use for data
    '''

    def __init__(self, cdf_resolution=10, measure='quantile_all', outcome_col='y'):
        self.cdf_resolution = cdf_resolution
        self.measure = measure
        self.outcome_col = outcome_col

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
        outcome_col = self.outcome_col
        n_firms = len(jids)

        ## Initialize cdf array ##
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

        # Group by income cdfs
        if measure == 'quantile_all':
            # Convert columns to NumPy
            j = frame.loc[:, 'j'].to_numpy()
            y = frame.loc[:, outcome_col].to_numpy()
            w = frame.loc[:, 'row_weights'].to_numpy()

            # Force j and jids to be integers so np.bincount and indexing work correctly
            if jids.dtype == 'O':
                jids = jids.astype(int, copy=True)
            if j.dtype == 'O':
                j = j.astype(int, copy=True)

            # Get quantiles from all data
            quantile_groups = DescrStatsW(y, weights=w).quantile(quantiles, return_pandas=False)

            ## Generate firm-level cdfs ##
            for i, quant in enumerate(quantile_groups):
                # Weighted number of observations at or below quantile
                cdfs[:, i] = np.bincount(j, w * (y <= quant))[jids]

            # Normalize by firm size (convert to cdf)
            jsize = np.bincount(j, w)[jids]
            cdfs = (cdfs.T / jsize.T).T

        elif measure == 'quantile_firm':
            # Sort frame by firm + compensation (do this once now, so that don't need to do it again later) (also note it is faster to sort and then manually compute quantiles than to use built-in quantile functions)
            # NOTE: don't sort in-place, otherwise modifies external data
            frame = frame.sort_values(['j', outcome_col], inplace=False)
            frame.reset_index(drop=True, inplace=True)
            frame.reset_index(drop=False, inplace=True)

            # Convert columns to NumPy (after sorting)
            y = frame.loc[:, outcome_col].to_numpy()
            w = frame.loc[:, 'row_weights'].to_numpy()

            # Find min + max index for each firm
            groupby_j = frame.groupby('j', sort=False)['index']
            j_min_idx = groupby_j.min().to_numpy()
            j_max_idx = groupby_j.max().to_numpy()
            del groupby_j

            ## Generate the cdfs ##
            for j in range(n_firms):
                # Get the firm-level compensation data (don't need to sort because already sorted)
                y_j = y[j_min_idx[j]: j_max_idx[j] + 1]
                w_j = w[j_min_idx[j]: j_max_idx[j] + 1]

                # Cumulative weight
                cum_w = w_j.cumsum()

                # Weighted number of observations
                weighted_n = w_j.sum()

                for q, quantile in enumerate(quantiles):
                    ## Generate the firm-level cdf ##
                    # Income index at particular quantile
                    idx = 0
                    for cum_w_val in cum_w:
                        if cum_w_val / weighted_n <= quantile:
                            idx += 1
                        else:
                            break
                    # Update cdfs with the firm-level cdf
                    cdfs[j, q] = y_j[min(idx, len(y_j) - 1)]

        return cdfs

class Moments:
    '''
    Generate compensation moments for firms. Used for clustering.

    Arguments:
        measures (str or list of str): how to compute the measures ('mean' to compute average income within each firm; 'var' to compute variance of income within each firm; 'max' to compute max income within each firm; 'min' to compute min income within each firm)
        outcome_col (str): outcome_col column to use for data
    '''

    def __init__(self, measures='mean', outcome_col='y'):
        self.measures = measures
        self.outcome_col = outcome_col

    def _compute_measure(self, frame, jids):
        '''
        Arguments:
            frame (Pandas DataFrame): data to use
            jids (list): sorted list of firm ids in frame (since frame could be a subset of self, this is not necessarily all firms in self)

        Returns:
            (NumPy Array): NumPy array of firm moments
        '''
        n_firms = len(jids)
        measures = self.measures
        n_measures = len(to_list(measures))
        outcome_col = self.outcome_col

        ## Initialize moments array ##
        moments = np.zeros([n_firms, n_measures])

        # Required for aggregate_transform
        # NOTE: don't sort in-place, otherwise modifies external data
        frame = frame.sort_values('j', inplace=False)

        for j, measure in enumerate(to_list(measures)):
            if measure == 'mean':
                # Group by mean income
                j_ = frame.loc[:, 'j'].to_numpy()
                y = frame.loc[:, outcome_col].to_numpy()
                w = frame.loc[:, 'row_weights'].to_numpy()
                # Force j and jids to be integers so np.bincount and indexing work correctly
                if jids.dtype == 'O':
                    jids = jids.astype(int, copy=True)
                if j_.dtype == 'O':
                    j_ = j_.astype(int, copy=True)
                moments[:, j] = np.bincount(j_, w * y)[jids] / np.bincount(j_, w)[jids]
                del j_, y, w
            elif measure == 'var':
                # Group by variance of income
                moments[:, j] = aggregate_transform(frame, 'j', outcome_col, 'var', weights='row_weights', merge=False)
            elif measure == 'max':
                moments[:, j] = frame.groupby('j', sort=False)[outcome_col].max().to_numpy()
            elif measure == 'min':
                moments[:, j] = frame.groupby('j', sort=False)[outcome_col].min().to_numpy()

        return moments
