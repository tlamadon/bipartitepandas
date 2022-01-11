'''
Functions for computing cluster groups
'''
import numpy as np
from sklearn.cluster import KMeans

class kmeans:
    '''
    Compute kmeans groups for data. Used for clustering.

    Arguments:
        **kwargs: parameters for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    '''

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def compute_groups(self, data, weights):
        '''
        Compute kmeans groups for data.

        Arguments:
            data (NumPy Array): data to group
            weights (NumPy Array or None): firm weights for clustering

        Returns:
            groups (NumPy Array): kmeans groups for data
        '''
        groups = KMeans(**self.kwargs).fit(data, sample_weight=weights).labels_
        return groups

class quantiles:
    '''
    Compute quantile groups for data. Used for clustering.

    Arguments:
        n_quantiles (int): number of quantiles to compute for groups
    '''

    def __init__(self, n_quantiles=4):
        self.n_quantiles = n_quantiles

    def compute_groups(self, data, weights):
        '''
        Compute quantiles groups for data.

        Arguments:
            data (NumPy Array): data to group
            weights (NumPy Array): required for consistent argument inputs with kmeans, not used in this function

        Returns:
            groups (NumPy Array): quantile groups for data
        '''
        n_quantiles = self.n_quantiles
        groups = np.zeros(shape=len(data))
        quantiles = np.linspace(1 / n_quantiles, 1, n_quantiles)
        quantile_groups = np.quantile(data, quantiles)
        for i, mean_income in enumerate(data):
            # Find quantile for each firm
            quantile_group = 0
            for quantile in quantile_groups:
                if mean_income > quantile:
                    quantile_group += 1
                else:
                    break
            groups[i] = quantile_group
        return groups
