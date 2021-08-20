'''
Class for a simulated two-way fixed effect network
'''
import numpy as np
from numpy import matlib
import pandas as pd
# from random import choices
from scipy.stats import norm
from scipy.linalg import eig
ax = np.newaxis
from bipartitepandas import update_dict, logger_init

class SimBipartite:
    '''
    Class of SimBipartite, where SimBipartite simulates a bipartite network of firms and workers.

    Arguments:
        sim_params (dict): parameters for simulated data

            Dictionary parameters:

                num_ind (int, default=10000): number of workers

                num_time (int, default=5): time length of panel

                firm_size (int, default=50): max number of individuals per firm

                nk (int, default=10): number of firm types

                nl (int, default=5): number of worker types

                alpha_sig (float, default=1): standard error of individual fixed effect (volatility of worker effects)

                psi_sig (float, default=1): standard error of firm fixed effect (volatility of firm effects)

                w_sig (float, default=1): standard error of residual in AKM wage equation (volatility of wage shocks)

                csort (float, default=1): sorting effect

                cnetw (float, default=1): network effect

                csig (float, default=1): standard error of sorting/network effects

                p_move (float, default=0.5): probability a worker moves firms in each period

                seed (int, default=None): NumPy RandomState seed
    '''

    def __init__(self, sim_params={}):
        # Start logger
        logger_init(self)
        # self.logger.info('initializing SimBipartite object')

        # Define default parameter dictionaries
        self.default_sim_params = {
            'num_ind': 10000, # Number of workers
            'num_time': 5, # Time length of panel
            'firm_size': 50, # Max number of individuals per firm
            'nk': 10, # Number of firm types
            'nl': 5, # Number of worker types
            'alpha_sig': 1, # Standard error of individual fixed effect (volatility of worker effects)
            'psi_sig': 1, # Standard error of firm fixed effect (volatility of firm effects)
            'w_sig': 1, # Standard error of residual in AKM wage equation (volatility of wage shocks)
            'csort': 1, # Sorting effect
            'cnetw': 1, # Network effect
            'csig': 1, # Standard error of sorting/network effects
            'p_move': 0.5, # Probability a worker moves firms in any period
            'seed': None # np.random.RandomState() seed
        }

        # Update parameters to include user parameters
        self.sim_params = update_dict(self.default_sim_params, sim_params)

        # Create NumPy Generator instance
        self.rng = np.random.default_rng(self.sim_params['seed'])

        # Prevent plotting unless results exist
        self.monte_carlo_res = False

    def __sim_network_gen_fe(self, sim_params):
        '''
        Generate fixed effects values for simulated panel data corresponding to the calibrated model.

        Arguments:
            sim_params (dict): parameters for simulated data

                Dictionary parameters:

                    num_ind (int, default=10000): number of workers

                    num_time (int, default=5): time length of panel

                    firm_size (int, default=50): max number of individuals per firm

                    nk (int, default=10): number of firm types

                    nl (int, default=5): number of worker types

                    alpha_sig (float, default=1): standard error of individual fixed effect (volatility of worker effects)

                    psi_sig (float, default=1): standard error of firm fixed effect (volatility of firm effects)

                    w_sig (float, default=1): standard error of residual in AKM wage equation (volatility of wage shocks)

                    csort (float, default=1): sorting effect

                    cnetw (float, default=1): network effect

                    csig (float, default=1): standard error of sorting/network effects

                    p_move (float, default=0.5): probability a worker moves firms in any period

        Returns:
            psi (NumPy Array): array of firm fixed effects
            alpha (NumPy Array): array of individual fixed effects
            G (NumPy Array): transition matrices
            H (NumPy Array): stationary distribution
        '''
        # Extract parameters
        nk, nl, alpha_sig, psi_sig = sim_params['nk'], sim_params['nl'], sim_params['alpha_sig'], sim_params['psi_sig']
        csort, cnetw, csig = sim_params['csort'], sim_params['cnetw'], sim_params['csig']

        # Draw fixed effects
        psi = norm.ppf(np.linspace(1, nk, nk) / (nk + 1)) * psi_sig
        alpha = norm.ppf(np.linspace(1, nl, nl) / (nl + 1)) * alpha_sig

        # Generate transition matrices
        G = norm.pdf((psi[ax, ax, :] - cnetw * psi[ax, :, ax] - csort * alpha[:, ax, ax]) / csig)
        G = np.divide(G, G.sum(axis=2)[:, :, ax])

        # Generate empty stationary distributions
        H = np.ones((nl, nk)) / nl

        # Solve stationary distributions
        for l in range(0, nl):
            # Solve eigenvectors
            # Source: https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain
            S, U = eig(G[l, :, :].T)
            stationary = np.array(U[:, np.where(np.abs(S-1.) < 1e-8)[0][0]].flat)
            stationary = stationary / np.sum(stationary)
            H[l, :] = stationary

        return psi, alpha, G, H

    def __sim_network_draw_fids(self, freq, num_time, firm_size):
        '''
        Draw firm ids for individual, given data that is grouped by worker id, spell id, and firm type.

        Arguments:
            freq (NumPy Array): size of groups (groups by worker id, spell id, and firm type)
            num_time (int): time length of panel
            firm_size (int): max number of individuals per firm

        Returns:
            (NumPy Array): random firms for each group
        '''
        max_int = int(np.maximum(1, freq.sum() / (firm_size * num_time)))
        return np.array(self.rng.choice(max_int, size=freq.count()) + 1)

    def sim_network(self):
        '''
        Simulate panel data corresponding to the calibrated model.

        Returns:
            data (Pandas DataFrame): simulated network
        '''
        # Generate fixed effects
        psi, alpha, G, H = self.__sim_network_gen_fe(self.sim_params)

        # Extract parameters
        num_ind, num_time, firm_size = self.sim_params['num_ind'], self.sim_params['num_time'], self.sim_params['firm_size']
        nk, nl, w_sig, p_move = self.sim_params['nk'], self.sim_params['nl'], self.sim_params['w_sig'], self.sim_params['p_move']

        # Generate empty NumPy arrays
        network = np.zeros((num_ind, num_time), dtype=int)
        spellcount = np.ones((num_ind, num_time))

        # Random draws of worker types for all individuals in panel
        sim_worker_types = self.rng.integers(low=0, high=nl, size=num_ind)

        for i in range(0, num_ind):
            l = sim_worker_types[i]
            # At time 1, we draw from H for initial firm
            network[i, 0] = self.rng.choice(range(0, nk), p=H[l, :]) # choices(range(0, nk), H[l, :])[0]

            for t in range(1, num_time):
                # Hit moving shock
                if self.rng.random() < p_move:
                    network[i, t] = self.rng.choice(range(0, nk), p=G[l, network[i, t - 1], :]) # choices(range(0, nk), G[l, network[i, t - 1], :])[0]
                    spellcount[i, t] = spellcount[i, t - 1] + 1
                else:
                    network[i, t] = network[i, t - 1]
                    spellcount[i, t] = spellcount[i, t - 1]

        # Compiling IDs and timestamps
        ids = np.reshape(np.outer(range(1, num_ind + 1), np.ones(num_time)), (num_time * num_ind, 1))
        ids = ids.astype(int)[:, 0]
        ts = np.reshape(np.matlib.repmat(range(1, num_time + 1), num_ind, 1), (num_time * num_ind, 1))
        ts = ts.astype(int)[:, 0]

        # Compiling worker types
        types = np.reshape(np.outer(sim_worker_types, np.ones(num_time)), (num_time * num_ind, 1))
        alpha_data = alpha[types.astype(int)][:, 0]

        # Compiling firm types
        psi_data = psi[np.reshape(network, (num_time * num_ind, 1))][:, 0]
        k_data = np.reshape(network, (num_time * num_ind, 1))[:, 0]

        # Compiling spell data
        spell_data = np.reshape(spellcount, (num_time * num_ind, 1))[:, 0]

        # Merging all columns into a dataframe
        data = pd.DataFrame(data={'i': ids, 't': ts, 'k': k_data,
                                'alpha': alpha_data, 'psi': psi_data,
                                'spell': spell_data.astype(int)})

        # Generate size of spells
        dspell = data.groupby(['i', 'spell', 'k']).size().to_frame(name='freq').reset_index()
        # Draw firm ids
        dspell['j'] = dspell.groupby(['k'])['freq'].transform(self.__sim_network_draw_fids, *[num_time, firm_size])
        # Make firm ids contiguous (and have them start at 1)
        dspell['j'] = dspell.groupby(['k', 'j'])['freq'].ngroup() + 1

        # Merge spells into panel
        data = data.merge(dspell, on=['i', 'spell', 'k'])

        data['move'] = (data['j'] != data['j'].shift(1)) & (data['i'] == data['i'].shift(1))

        # Compute wages through the AKM formula
        data['y'] = data['alpha'] + data['psi'] + w_sig * norm.rvs(size=num_ind * num_time, random_state=self.rng)

        data['i'] -= 1 # Start at 0
        data['j'] -= 1 # Start at 0

        return data
