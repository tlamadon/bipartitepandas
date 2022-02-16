'''
Classes for computing cluster groups. Note: use classes rather than nested functions because nested functions cannot be pickled (source: https://stackoverflow.com/a/12022055/17333120).
'''
import numpy as np
try:
    # Optimize sklearn (source: https://intel.github.io/scikit-learn-intelex/)
    from sklearnex import patch_sklearn
    patch_sklearn('KMeans')
except ImportError:
    pass
from sklearn.cluster import KMeans as sklKMeans

class KMeans:
    '''
    Compute KMeans groups for data. Used for clustering.

    Arguments:
        **kwargs: parameters for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    '''

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def compute_groups(self, data, weights):
        '''
        Compute KMeans groups for data.

        Arguments:
            data (NumPy Array): data to group
            weights (NumPy Array or None): firm weights for clustering

        Returns:
            (NumPy Array): KMeans groups for data
        '''
        groups = sklKMeans(**self.kwargs).fit(data, sample_weight=weights).labels_
        return groups

class Quantiles:
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
            weights (NumPy Array): required for consistent argument inputs with KMeans, not used in this function

        Returns:
            (NumPy Array): quantile groups for data
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
