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
        **kwargs: parameters for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html); note that key 'random_state' will be overwritten
    '''

    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()

        if 'random_state' in self.kwargs.keys():
            # Remove 'random_state' key
            del self.kwargs['random_state']

    def _compute_groups(self, data, weights, rng=None):
        '''
        Compute KMeans groups for data.

        Arguments:
            data (NumPy Array): data to group
            weights (NumPy Array or None): firm weights for clustering
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (NumPy Array): KMeans groups for data
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        if isinstance(rng, np.random._generator.Generator):
            # Generate seed - SKLearn is not compatible with np.random.default_rng
            random_state = rng.bit_generator._seed_seq.spawn(1)[0].generate_state(1)[0]
        else:
            random_state = rng

        groups = sklKMeans(random_state=random_state, **self.kwargs).fit(data, sample_weight=weights).labels_
        return groups

class Quantiles:
    '''
    Compute quantile groups for data. Used for clustering.

    Arguments:
        n_quantiles (int): number of quantiles to compute for groups
    '''

    def __init__(self, n_quantiles=4):
        self.n_quantiles = n_quantiles

    def _compute_groups(self, data, weights, rng=None):
        '''
        Compute quantiles groups for data.

        Arguments:
            data (NumPy Array): data to group
            weights (NumPy Array): used for KMeans, not used for Quantiles
            rng (np.random.Generator or None): used for KMeans, not used for Quantiles

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
