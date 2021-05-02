'''
Functions for computing cluster groups
'''
import numpy as np
from sklearn.cluster import KMeans

def kmeans(**kwargs):
    '''
    Compute kmeans groups for data. Used for clustering.

    Arguments:
        **kwargs: parameters for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

    Returns:
        compute_kmeans (function): subfunction
    '''
    # Workaround for multiprocessing
    # Source: https://stackoverflow.com/a/61879723
    global compute_kmeans

    def compute_kmeans(data, weights):
        '''
        Compute kmeans groups for data.

        Arguments:
            data (NumPy Array): data to group
            weights (NumPy Array or None): firm weights for clustering

        Returns:
            groups (NumPy Array): kmeans groups for data
        '''
        groups = KMeans(**kwargs).fit(data, sample_weight=weights).labels_
        return groups
    return compute_kmeans

def quantiles(n_quantiles=4):
    '''
    Compute quantile groups for data. Used for clustering.

    Arguments:
        n_quantiles (int): number of quantiles to compute for groups

    Returns:
        compute_quantiles (function): subfunction
    '''
    # Workaround for multiprocessing
    # Source: https://stackoverflow.com/a/61879723
    global compute_quantiles

    def compute_quantiles(data, weights):
        '''
        Compute quantiles groups for data.

        Arguments:
            data (NumPy Array): data to group
            weights (NumPy Array): required for consistent argument inputs with kmeans, not used in this function

        Returns:
            groups (NumPy Array): quantile groups for data
        '''
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
    return compute_quantiles
