'''
Class for simulating bipartite networks.
'''
import numpy as np
ax = np.newaxis
from pandas import DataFrame
from scipy.stats import norm
from bipartitepandas.util import ParamsDict

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return a >= 0
def _gt0(a):
    return a > 0
def _0to1(a):
    return 0 <= a <= 1

# Define default parameter dictionary
_sim_params_default = ParamsDict({
    'n_workers': (10000, 'type_constrained', (int, _gteq1),
        '''
            (default=10000) Number of workers.
        ''', '>= 1'),
    'n_time': (5, 'type_constrained', (int, _gteq1),
        '''
            (default=5) Time length of panel.
        ''', '>= 1'),
    'firm_size': (50, 'type_constrained', ((float, int), _gt0),
        '''
            (default=50) Average observations per firm per time period.
        ''', '> 0'),
    'nk': (10, 'type_constrained', (int, _gteq1),
        '''
            (default=10) Number of firm types.
        ''', '>= 1'),
    'nl': (5, 'type_constrained', (int, _gteq1),
        '''
            (default=5) Number of worker types.
        ''', '>= 1'),
    'alpha_sig': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Standard error of individual fixed effect (volatility of worker effects).
        ''', '>= 0'),
    'psi_sig': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Standard error of firm fixed effect (volatility of firm effects).
        ''', '>= 0'),
    'w_sig': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Standard error of residual in AKM wage equation (volatility of wage shocks).
        ''', '>= 0'),
    'c_sort': (1, 'type', (float, int),
        '''
            (default=1) Sorting effect.
        ''', None),
    'c_netw': (1, 'type', (float, int),
        '''
            (default=1) Network effect.
        ''', None),
    'c_sig': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Standard error of sorting/network effects.
        ''', '>= 0'),
    'p_move': (0.5, 'type_constrained', ((float, int), _0to1),
        '''
            (default=0.5) Probability a worker moves firms in any period.
        ''', 'in [0, 1]')
})

def sim_params(update_dict=None):
    '''
    Dictionary of default sim_params. Run bpd.sim_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict or None): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict): dictionary of sim_params
    '''
    new_dict = _sim_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

class SimBipartite:
    '''
    Class of SimBipartite, where SimBipartite simulates a bipartite network of firms and workers.

    Arguments:
        params (ParamsDict or None): dictionary of parameters for simulating data. Run bpd.sim_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.sim_params().
    '''

    def __init__(self, params=None):
        if params is None:
            params = sim_params()

        # Store parameters
        self.params = params

    def _gen_fe(self):
        '''
        Generate fixed effects values for simulated panel data corresponding to the calibrated model.

        Returns:
            psi (NumPy Array): array of firm fixed effects
            alpha (NumPy Array): array of individual fixed effects
            G (NumPy Array): transition matrices
            H (NumPy Array): stationary distribution
        '''
        # Extract parameters
        nk, nl, alpha_sig, psi_sig, c_sort, c_netw, c_sig = self.params.get_multiple(('nk', 'nl', 'alpha_sig', 'psi_sig', 'c_sort', 'c_netw', 'c_sig'))

        # Draw fixed effects
        psi = norm.ppf(np.linspace(1, nk, nk) / (nk + 1)) * psi_sig
        alpha = norm.ppf(np.linspace(1, nl, nl) / (nl + 1)) * alpha_sig

        # Generate transition matrices
        G = norm.pdf((psi[ax, ax, :] - c_netw * psi[ax, :, ax] - c_sort * alpha[:, ax, ax]) / c_sig)
        G = G / G.sum(axis=2)[:, :, ax]

        # Generate empty stationary distributions
        H = np.ones((nl, nk)) / nl

        # Solve stationary distributions
        for l in range(nl):
            # Solve eigenvectors (source: https://stackoverflow.com/a/58334399/17333120)
            evals, evecs = np.linalg.eig(G[l, :, :].T)
            stationary = evecs[:, np.isclose(evals, 1)][:, 0]
            # Take real component
            stationary = np.real(stationary)
            # Normalize
            stationary = stationary / np.sum(stationary)
            H[l, :] = stationary

        return psi, alpha, G, H

    def _draw_fids(self, freq, rng=None):
        '''
        Draw firm ids for a particular firm type.

        Arguments:
            freq (NumPy Array): number of observations in each spell that occurs at the given firm type
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (NumPy Array): random firm ids for each spell that occurs at the given firm type
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Set the maximum firm id that can be drawn such that the average number of observations per firm per time period is approximately 'firm_size'
        max_firm_id = max(1, round(freq.sum() / (self.params['firm_size'] * self.params['n_time'])))
        return rng.choice(max_firm_id, size=freq.count())

    def simulate(self, rng=None):
        '''
        Simulate panel data corresponding to the calibrated model. Columns are as follows: i=worker id; j=firm id; y=wage; t=time period; l=worker type; k=firm type; alpha=worker effect; psi=firm effect.

        Arguments:
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): simulated network
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        n_workers, n_time, nk, nl, w_sig, p_move = self.params.get_multiple(('n_workers', 'n_time', 'nk', 'nl', 'w_sig', 'p_move'))

        # Generate fixed effects
        psi, alpha, G, H = self._gen_fe()

        # Generate empty NumPy arrays
        network = np.zeros((n_workers, n_time), dtype=int)
        spellcount = np.zeros((n_workers, n_time), dtype=int)

        # Random draws of worker types for all individuals in panel
        sim_worker_types = rng.integers(low=0, high=nl, size=n_workers)

        for i in range(n_workers):
            l = sim_worker_types[i]
            # At time 0, we draw from H for initial firm
            network[i, 0] = rng.choice(range(nk), p=H[l, :])
            spellcount[i, 0] = spellcount[i - 1, n_time - 1] + 1

            for t in range(1, n_time):
                if rng.random() < p_move:
                    # Move firms
                    network[i, t] = rng.choice(range(nk), p=G[l, network[i, t - 1], :])
                    spellcount[i, t] = spellcount[i, t - 1] + 1
                else:
                    # Stay at the same firm
                    network[i, t] = network[i, t - 1]
                    spellcount[i, t] = spellcount[i, t - 1]

        # Compile ids and timestamps
        ids = np.repeat(range(n_workers), n_time)
        ts = np.tile(range(n_time), n_workers)

        # Compile worker types
        l_data = np.repeat(sim_worker_types, n_time)
        alpha_data = alpha[l_data]

        # Compile firm types
        k_data = network.flatten()
        psi_data = psi[k_data]

        # Compile spell data
        spell_data = spellcount.flatten()

        # Merge all columns into a dataframe
        data = DataFrame(data={'i': ids, 't': ts, 'l': l_data, 'k': k_data,
                                'alpha': alpha_data, 'psi': psi_data,
                                'spell': spell_data})

        # Generate size of spells
        dspell = data.groupby(['spell', 'k'], sort=False).size().to_frame(name='freq')
        dspell.reset_index(inplace=True)
        # Draw firm ids
        dspell.loc[:, 'j'] = dspell.groupby('k')['freq'].transform(self._draw_fids, rng)
        # Make firm ids contiguous
        dspell.loc[:, 'j'] = dspell.groupby(['k', 'j'])['freq'].ngroup()

        # Merge firm ids into panel
        data = data.merge(dspell.loc[:, ['spell', 'j']], on='spell')
        # spell_to_firm = dspell.groupby('spell')['j'].first().to_dict()
        # data.loc[:, 'j'] = data.loc[:, 'spell'].map(spell_to_firm)

        # data['move'] = (data['j'] != data['j'].shift(1)) & (data['i'] == data['i'].shift(1))

        # Compute wages through the AKM formula
        data.loc[:, 'y'] = data.loc[:, 'alpha'] + data.loc[:, 'psi'] + w_sig * rng.normal(size=n_workers * n_time)

        return data.reindex(['i', 'j', 'y', 't', 'l', 'k', 'alpha', 'psi'], axis=1, copy=False)
