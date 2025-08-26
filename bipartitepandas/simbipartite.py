'''
Class for simulating bipartite networks.
'''
from paramsdict import ParamsDict
from numba import njit
import numpy as np
from pandas import DataFrame
from scipy.stats import norm

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
sim_params = ParamsDict({
    'nk': (10, 'type_constrained', (int, _gteq1),
        '''
            (default=10) Number of firm classes.
        ''', '>= 1'),
    'nl': (5, 'type_constrained', (int, _gteq1),
        '''
            (default=5) Number of worker types.
        ''', '>= 1'),
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
    'firm_size_distn': ('normal', 'set', ['normal', 'pareto', 'exponential'],
        '''
            (default='normal') Firm size distribution. If 'normal', firm sizes are normally distributed. If 'pareto', firm sizes are Pareto distributed. If 'exponential', firm sizes are exponentially distributed.
        ''', None),
    'max_firm_size': (500, 'type_constrained', ((float, int), _gt0),
        '''
            (default=500) Approximate maximum number of workers per firm per time period, used with Pareto firm size distribution.
        ''', '> 0'),
    'exp_distn_exponent': (1.5, 'type_constrained', ((float, int), _gt0),
        '''
            (default=1.5) Exponential distribution exponent.
        ''', '> 0'),
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
            (default=1) Sorting effect. Controls sorting between workers and firms based on similarity of worker FE and firm FE. If c_sort=0, sorting is random. If c_sort>0, sorting is positive. If c_sort<0, sorting is negative.
        ''', None),
    'c_netw': (1, 'type', (float, int),
        '''
            (default=1) Network effect. Controls mobility patterns between firms based on similarity of firm FE. If c_netw=0, mobility is uncorrelated with firm FE similarity. If c_netw=1, maximizes mobility between firms with similar firm FE.
        ''', None),
    'c_sig': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Scale of sorting/network effects. As c_sig goes to infinity, the effect of c_sort and c_netw on sorting and mobility patterns goes to 0, and sorting/mobility become random.
        ''', '>= 0'),
    'p_move': (0.5, 'type_constrained', ((float, int), _0to1),
        '''
            (default=0.5) Probability a worker moves firms in any period.
        ''', 'in [0, 1]')
})

@njit
def random_choice(a, p, rng):
    '''
    Source: https://github.com/numba/numba/issues/2539#issuecomment-2214103183
    :param a: Sample values
    :param p: Probabilities of each sample
    :param rng: A Numpy random number generator instance
    :return: A random sample of a
    '''
    rand = rng.random()
    cdf = 0.0
    for i in range(len(a)):
        cdf += p[i]
        if rand <= cdf:
            return a[i]
    return len(a) - 1

def _inv_cdf_trunc_pareto(u, alpha, xm=1.0, x_max=None):
    '''
    Inverse CDF for a Pareto(xm, alpha) truncated at x_max (>= xm).
    If x_max is None, returns standard Pareto inverse: xm * (1-u)^(-1/alpha).
    For truncated Pareto:
      F_tr(x) = (F(x) - F(xm)) / (F(x_max) - F(xm)) with F(xm)=0 if xm is minimum
    Solve u = F_tr(x) => x = xm * (1 - u*(1 - (xm/x_max)^alpha))^(-1/alpha)
    '''
    if x_max is None:
        return xm * (1.0 - u) ** (-1.0 / alpha)
    # Precompute (xm/x_max)^alpha
    c = (xm / x_max) ** alpha
    # Inverse of truncated CDF:
    # u in [0,1] => x = xm * (1 - u*(1 - c))^(-1/alpha)
    return xm * (1.0 - u * (1.0 - c)) ** (-1.0 / alpha)

@njit
def _draw_with_capacities(ids, freq, capacities, rng):
    # Assign each spell to a firm with probability ∝ remaining capacity
    firm_id_lst = np.arange(len(capacities))
    firm_probs = capacities / np.sum(capacities)
    for idx, f in enumerate(freq):
        total_remaining = np.sum(capacities)
        if total_remaining <= 0:
            # Fallback to the original proportions if rounding emptied capacities
            p = firm_probs
        else:
            p = capacities / total_remaining
        j = random_choice(firm_id_lst, p, rng)
        ids[idx] = j
        capacities[j] = max(0.0, capacities[j] - f)

@njit
def _sim_network(network, spellcount, worker_types, nk, nl, p_move, G, H, rng):
    '''
    Simulate mobility network.

    Arguments:
        network (NumPy Array):
                mobility network
        spellcount (NumPy Array):
                matrix of mobility spells
        worker_types (NumPy Array):
                vector of worker types
        nk (int):
                number of firm classes
        nl (int):
                number of worker types
        p_move (float):
                probability a worker moves firms in any period
        G (NumPy Array):
                worker type-firm class transition matrix
        H (NumPy Array):
                worker type-firm class stationary distribution
        rng (np.random.Generator):
                NumPy random number generator; None is equivalent to np.random.default_rng(None)
    '''
    n_workers, n_time = network.shape

    # Lists
    nk_lst  = np.arange(nk)
    t_lst   = np.arange(1, n_time)

    for i in range(n_workers):
        # Worker type
        l = (i % nl)
        worker_types[i] = l

        # At time 0, we draw from H for initial firm
        network[i, 0] = random_choice(nk_lst, H[l, :], rng)
        if i > 0:
            spellcount[i, 0] = spellcount[i - 1, n_time - 1] + 1

        for t in t_lst:
            if rng.random() < p_move:
                # Move firms
                network[i, t]       = random_choice(nk_lst, G[l, network[i, t - 1], :], rng)
                spellcount[i, t]    = spellcount[i, t - 1] + 1
            else:
                # Stay at the same firm
                network[i, t]       = network[i, t - 1]
                spellcount[i, t]    = spellcount[i, t - 1]

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
        nk, nl, alpha_sig, psi_sig, c_sort, c_netw, c_sig \
            = self.params.get_multiple((
                    'nk', 'nl', 'alpha_sig', 'psi_sig', 'c_sort', 'c_netw', 'c_sig'
                ))

        # Draw fixed effects
        psi = norm.ppf(np.linspace(1, nk, nk) / (nk + 1)) * psi_sig
        alpha = norm.ppf(np.linspace(1, nl, nl) / (nl + 1)) * alpha_sig

        # Generate transition matrices
        G = norm.pdf((psi[None, None, :] - c_netw * psi[None, :, None]
                            - c_sort * alpha[:, None, None]) / c_sig)
        G = G / G.sum(axis=2, keepdims=True)

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
            freq (NumPy Array):
                    number of observations in each spell that occurs at the given firm type
            rng (np.random.Generator):
                    NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (NumPy Array): random firm ids for each spell that occurs at the given firm type
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        n_time, firm_size, firm_size_distn, max_firm_size, exp_distn_exponent \
            = self.params.get_multiple((
                    'n_time', 'firm_size', 'firm_size_distn', 'max_firm_size', 'exp_distn_exponent'
                ))

        # Set the maximum firm id that can be drawn such that the average number of observations per firm per time period is approximately 'firm_size'
        n_obs           = freq.sum()
        obs_per_firm    = n_time * firm_size
        n_firms         = max(1, round(n_obs / obs_per_firm))
        if firm_size_distn == 'normal':
            return rng.choice(n_firms, size=len(freq))
        elif firm_size_distn == 'pareto':
            # Back out alpha from mean=firm_size
            alpha = firm_size / (firm_size - 1.0)

            # Quantiles
            q_mid = (np.arange(1, n_firms + 1) - 0.5) / n_firms

            # Truncated Pareto quantiles
            firm_sizes = _inv_cdf_trunc_pareto(q_mid, alpha, xm=1.0, x_max=max_firm_size)

        elif firm_size_distn == 'exponential':
            firm_sizes = np.exp(exp_distn_exponent * np.log(np.linspace(1, 100, n_firms)))

        firm_probs = firm_sizes / firm_sizes.sum()
        capacities = firm_probs * n_obs

        # Capacity-proportional assignment of spells to firms
        # Target capacity (in *observations*, not spells) for firm j
        ids = np.zeros(len(freq), dtype=int)
        _draw_with_capacities(ids, freq.to_numpy(), capacities, rng)

        return ids

    def simulate(self, rng=None):
        '''
        Simulate panel data corresponding to the calibrated model. Columns are as follows: i=worker id; j=firm id; y=wage; t=time period; l=worker type; k=firm type; alpha=worker effect; psi=firm effect.

        Arguments:
            rng (np.random.Generator):
                    NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): simulated network
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        n_workers, n_time, nk, nl, w_sig, p_move \
            = self.params.get_multiple((
                    'n_workers', 'n_time', 'nk', 'nl', 'w_sig', 'p_move'
                ))

        # Generate fixed effects
        psi, alpha, G, H = self._gen_fe()

        # Simulate mobility network
        # Generate empty NumPy arrays
        network         = np.zeros((n_workers, n_time), dtype=int)
        spellcount      = np.zeros((n_workers, n_time), dtype=int)
        worker_types    = np.zeros(n_workers, dtype=int)
        _sim_network(network, spellcount, worker_types, nk, nl, p_move, G, H, rng)

        # Compile ids and timestamps
        ids = np.repeat(np.arange(n_workers), n_time)
        ts = np.tile(np.arange(n_time), n_workers)

        # Compile worker types
        l_data = np.repeat(worker_types, n_time)
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

        # Compute wages through the AKM formula
        data.loc[:, 'y'] = data.loc[:, 'alpha'] + data.loc[:, 'psi'] \
                                + w_sig * rng.normal(size=(n_workers * n_time))

        return data.reindex(['i', 'j', 'y', 't', 'l', 'k', 'alpha', 'psi'], axis=1, copy=False)
