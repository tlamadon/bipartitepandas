'''
Class for a simulated two-way fixed effect network
'''
import numpy as np
import pandas as pd
# from random import choices
from scipy.stats import norm
ax = np.newaxis
from bipartitepandas import logger_init, ParamsDict

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return a >= 0
def _0to1(a):
    return 0 <= a <= 1

# Define default parameter dictionary
_sim_params_default = ParamsDict({
    'num_ind': (10000, 'type_constrained', (int, _gteq1),
        '''
            (default=10000) Number of workers.
        ''', '>= 1'),
    'num_time': (5, 'type_constrained', (int, _gteq1),
        '''
            (default=5) Time length of panel.
        ''', '>= 1'),
    'firm_size': (50, 'type_constrained', (int, _gteq1),
        '''
            (default=50) Maximum number of individuals per firm.
        ''', '>= 1'),
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
    'csort': (1, 'type', (float, int),
        '''
            (default=1) Sorting effect.
        ''', None),
    'cnetw': (1, 'type', (float, int),
        '''
            (default=1) Network effect.
        ''', None),
    'csig': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Standard error of sorting/network effects.
        ''', '>= 0'),
    'p_move': (0.5, 'type_constrained', ((float, int), _0to1),
        '''
            (default=0.5) Probability a worker moves firms in any period.
        ''', 'in [0, 1]')
})

def sim_params(update_dict={}):
    '''
    Dictionary of default sim_params.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of sim_params
    '''
    new_dict = _sim_params_default.copy()
    new_dict.update(update_dict)
    return new_dict

class SimBipartite:
    '''
    Class of SimBipartite, where SimBipartite simulates a bipartite network of firms and workers.

    Arguments:
        sim_params (ParamsDict): dictionary of parameters for simulating data. Run bpd.sim_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, sim_params=sim_params()):
        # Start logger
        logger_init(self)
        # self.log('initializing SimBipartite object', level='info')

        # Store parameters
        self.sim_params = sim_params

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
        nk, nl, alpha_sig, psi_sig, csort, cnetw, csig = self.sim_params.get_multiple(('nk', 'nl', 'alpha_sig', 'psi_sig', 'csort', 'cnetw', 'csig'))

        # Draw fixed effects
        psi = norm.ppf(np.linspace(1, nk, nk) / (nk + 1)) * psi_sig
        alpha = norm.ppf(np.linspace(1, nl, nl) / (nl + 1)) * alpha_sig

        # Generate transition matrices
        G = norm.pdf((psi[ax, ax, :] - cnetw * psi[ax, :, ax] - csort * alpha[:, ax, ax]) / csig)
        G = G / G.sum(axis=2)[:, :, ax]

        # Generate empty stationary distributions
        H = np.ones((nl, nk)) / nl

        # Solve stationary distributions
        for l in range(nl):
            # Solve eigenvectors (source: https://stackoverflow.com/a/58334399/17333120)
            evals, evecs = np.linalg.eig(G[l, :, :].T)
            stationary = evecs[:, np.isclose(evals, 1)][:, 0]
            stationary = stationary / np.sum(stationary)
            H[l, :] = stationary

        return psi, alpha, G, H

    def _draw_fids(self, freq, rng=np.random.default_rng(None)):
        '''
        Draw firm ids for individual spells, setting the maximum firm id computing how many observations each firm type has.

        Arguments:
            freq (NumPy Array): size of groups (groups by worker id, spell id, and firm type)
            rng (np.random.Generator): NumPy random number generator

        Returns:
            (NumPy Array): random firms for each group
        '''
        max_int = int(np.maximum(1, freq.sum() / (self.sim_params['firm_size'] * self.sim_params['num_time'])))
        return rng.choice(max_int, size=freq.count())

    def sim_network(self, rng=np.random.default_rng(None)):
        '''
        Simulate panel data corresponding to the calibrated model.

        Arguments:
            rng (np.random.Generator): NumPy random number generator

        Returns:
            data (Pandas DataFrame): simulated network
        '''
        # Extract parameters
        num_ind, num_time, nk, nl, w_sig, p_move = self.sim_params.get_multiple(('num_ind', 'num_time', 'nk', 'nl', 'w_sig', 'p_move'))

        # Generate fixed effects
        psi, alpha, G, H = self._gen_fe()

        # Generate empty NumPy arrays
        network = np.zeros((num_ind, num_time), dtype=int)
        spellcount = np.zeros((num_ind, num_time), dtype=int)

        # Random draws of worker types for all individuals in panel
        sim_worker_types = rng.integers(low=0, high=nl, size=num_ind)

        for i in range(num_ind):
            l = sim_worker_types[i]
            # At time 1, we draw from H for initial firm
            network[i, 0] = rng.choice(range(nk), p=H[l, :])
            spellcount[i, 0] = spellcount[i - 1, num_time - 1] + 1

            for t in range(1, num_time):
                # Hit moving shock
                if rng.random() < p_move:
                    network[i, t] = rng.choice(range(nk), p=G[l, network[i, t - 1], :])
                    spellcount[i, t] = spellcount[i, t - 1] + 1
                else:
                    network[i, t] = network[i, t - 1]
                    spellcount[i, t] = spellcount[i, t - 1]

        # Compiling IDs and timestamps
        ids = np.repeat(range(num_ind), num_time)
        ts = np.tile(range(num_time), num_ind)

        # Compiling worker types
        l_data = np.repeat(sim_worker_types, num_time)
        alpha_data = alpha[l_data]

        # Compiling firm types
        k_data = network.flatten()
        psi_data = psi[k_data]

        # Compiling spell data
        spell_data = spellcount.flatten()

        # Merging all columns into a dataframe
        data = pd.DataFrame(data={'i': ids, 't': ts, 'k': k_data,
                                'alpha': alpha_data, 'psi': psi_data,
                                'spell': spell_data})

        # Generate size of spells
        dspell = data.groupby(['spell', 'k'], sort=False).size().to_frame(name='freq')
        dspell.reset_index(inplace=True)
        # Draw firm ids
        dspell.loc[:, 'j'] = dspell.groupby('k')['freq'].transform(self._draw_fids, rng)
        # Make firm ids contiguous
        dspell.loc[:, 'j'] = dspell.groupby(['k', 'j'])['freq'].ngroup()

        # Merge spells into panel
        data = data.merge(dspell.loc[:, ['spell', 'j']], on='spell')
        # spell_to_firm = dspell.groupby('spell')['j'].first().to_dict()
        # data.loc[:, 'j'] = data.loc[:, 'spell'].map(spell_to_firm)

        # data['move'] = (data['j'] != data['j'].shift(1)) & (data['i'] == data['i'].shift(1))

        # Compute wages through the AKM formula
        data.loc[:, 'y'] = data.loc[:, 'alpha'] + data.loc[:, 'psi'] + w_sig * rng.normal(size=num_ind * num_time)

        return data.loc[:, ['i', 'j', 'y', 't', 'k', 'alpha', 'psi', 'spell']]
