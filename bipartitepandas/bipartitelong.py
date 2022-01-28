'''
Class for a bipartite network in long form
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import warnings

# Define default parameter dictionary
_es_extended_plot_params_default = bpd.ParamsDict({
    'title_height': (1.0, 'type', float,
        '''
            (default=1.0) Location of titles for subfigures.
        '''),
    'fontsize': (9.0, 'type', float,
        '''
            (default=9.0) Font size of titles for subfigures.
        '''),
    'sharex': (True, 'type', bool,
        '''
            (default=True) Share x axis between subfigures.
        '''),
    'sharey': (True, 'type', bool,
        '''
            (default=True) Share y axis between subfigures.
        '''),
    'yticks_round': (1, 'type', int,
        '''
            (default=1) How many digits to round y ticks.
        ''')
})

def es_extended_plot_params(update_dict={}):
    '''
    Dictionary of default es_extended_plot_params.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of es_extended_plot_params
    '''
    new_dict = _es_extended_plot_params_default.copy()
    new_dict.update(update_dict)
    return new_dict

class BipartiteLong(bpd.BipartiteLongBase):
    '''
    Class for bipartite networks of firms and workers in long form. Inherits from BipartiteLongBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict or None): make data columns readable (requires: i (worker id), j (firm id), y (compensation), t (period); optionally include: g (firm cluster), m (0 if stayer, 1 if mover)). Keep None if column names already correct
        include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, include_id_reference_dict=False, **kwargs):
        # Initialize DataFrame
        reference_dict = {'t': 't'}
        super().__init__(*args, reference_dict=reference_dict, col_dict=col_dict, include_id_reference_dict=include_id_reference_dict, **kwargs)

        # self.log('BipartiteLong object initialized', level='info')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteLong

    @property
    def _constructor_es(self):
        '''
        For get_es(), tells BipartiteLongBase which event study format to use.

        Returns:
            (BipartiteEventStudy): class
        '''
        return bpd.BipartiteEventStudy

    def get_collapsed_long(self, is_sorted=False, copy=True):
        '''
        Collapse long data by job spells (so each spell for a particular worker at a particular firm is one observation).

        Arguments:
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            collapsed_frame (BipartiteLongCollapsed): BipartiteLongCollapsed object generated from long data collapsed by job spells
        '''
        self.log('beginning collapse', level='info')

        # Sort data by i (and t, if included)
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)
        self.log('data sorted by i (and t, if included)', level='info')

        # Generate spell ids
        spell_ids = frame._get_spell_ids(is_sorted=True, copy=False)

        # Quickly check whether a collapse is necessary
        if (len(frame) < 2) or (spell_ids[-1] == len(frame)):
            return frame

        ## Aggregate at the spell level
        spells = frame.groupby(spell_ids, sort=False)

        # First, prepare required columns for aggregation
        agg_funcs = {
            'i': pd.NamedAgg(column='i', aggfunc='first'),
            'j': pd.NamedAgg(column='j', aggfunc='first'),
            'y': pd.NamedAgg(column='y', aggfunc='mean'),
            'w': pd.NamedAgg(column='i', aggfunc='size')
        }

        # Next, prepare the time column for aggregation
        if self._col_included('t'):
            agg_funcs['t1'] = pd.NamedAgg(column='t', aggfunc='min')
            agg_funcs['t2'] = pd.NamedAgg(column='t', aggfunc='max')

        # Next, prepare optional columns for aggregation
        all_cols = self._included_cols()
        for col in all_cols:
            if col in self.columns_opt:
                if self.col_dtype_dict[col] == 'int':
                    for subcol in bpd.to_list(self.reference_dict[col]):
                        agg_funcs[subcol] = pd.NamedAgg(column=subcol, aggfunc='first')
                if self.col_dtype_dict[col] == 'float':
                    for subcol in bpd.to_list(self.reference_dict[col]):
                        agg_funcs[subcol] = pd.NamedAgg(column=subcol, aggfunc='mean')

        # Finally, aggregate
        data_spell = spells.agg(**agg_funcs)

        # Sort columns
        sorted_cols = sorted(data_spell.columns, key=bpd.col_order)
        data_spell = data_spell.reindex(sorted_cols, axis=1, copy=False)
        data_spell.reset_index(drop=True, inplace=True)

        self.log('data aggregated at the spell level', level='info')

        collapsed_frame = bpd.BipartiteLongCollapsed(data_spell, log=frame._log_on_indicator)
        collapsed_frame._set_attributes(self, no_dict=True)

        # m can change from long to collapsed long
        collapsed_frame = collapsed_frame.gen_m(force=True, copy=False)

        # If no time column, then returns will be collapsed
        if not collapsed_frame._col_included('t'):
            collapsed_frame.no_returns = True

        return collapsed_frame

    def _drop_i_t_duplicates(self, how='max', is_sorted=False, copy=True):
        '''
        Keep only the highest paying job for i-t (worker-year) duplicates.

        Arguments:
            how (str): if 'max', keep max paying job; otherwise, take `how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `how` can take any option valid for a Pandas transform. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): dataframe that keeps only the highest paying job for i-t (worker-year) duplicates. If no t column(s), returns frame with no changes
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        # Check whether any observations are dropped
        prev_len = len(frame)

        if frame._col_included('t'):
            frame = frame.sort_rows(is_sorted=is_sorted, copy=False)
            # Temporarily disable warnings
            warnings.filterwarnings('ignore')
            if how not in ['max', max]:
                # Group by worker id, time, and firm id, and take `how` of compensation
                frame.loc[:, 'y'] = frame.groupby(['i', 't', 'j'])['y'].transform(how)
            # Take max over duplicates
            # Source: https://stackoverflow.com/questions/23394476/keep-other-columns-when-doing-groupby
            frame = frame.loc[frame.loc[:, 'y'].to_numpy() == frame.groupby(['i', 't'], sort=False)['y'].transform(max).to_numpy(), :].groupby(['i', 't'], as_index=False, sort=False).first()
            # Restore warnings
            warnings.filterwarnings('default')

        # Data now has unique i-t observations
        frame.i_t_unique = True

        # If observations dropped, recompute 'm'
        if prev_len != len(frame):
            frame = frame.gen_m(force=True, copy=False)

        return frame

    def _construct_firm_worker_linkages(self, is_sorted=False):
        '''
        Construct numpy array linking firms to worker ids, for use with leave-one-observation-out components.

        Arguments:
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.

        Returns:
            (tuple of NumPy Array, int): (firm-worker linkages, maximum firm id)
        '''
        move_rows = (self.groupby('i', sort=(not is_sorted))['m'].transform('max') > 0)
        i_col = self.loc[move_rows, 'i'].to_numpy()
        j_col = self.loc[move_rows, 'j'].to_numpy()
        max_j = np.max(j_col)
        linkages = np.stack([i_col + max_j + 1, j_col], axis=1)

        return linkages, max_j

    def _get_articulation_obs(self, G, max_j, is_sorted=False):
        '''
        Compute articulation observations for self, by checking whether self is leave-one-observation-out connected when dropping selected observations one at a time.

        Arguments:
            G (igraph Graph): graph linking firms by movers
            max_j (int): maximum j
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.

        Returns:
            (NumPy Array): indices of articulation observations
        '''
        # Find bridges (recall i is adjusted to be greater than j, which is why we reverse the order) (source: https://igraph.discourse.group/t/function-to-find-edges-which-are-bridges-in-r-igraph/154/2)
        bridges = [tuple(sorted(a, reverse=True)) for a in G.biconnected_components() if len(a) == 2]
        bridges_workers = set([bridge[0] - (max_j + 1) for bridge in bridges])
        bridges_firms = set([bridge[1] for bridge in bridges])

        # Get possible articulation observations
        possible_articulation_obs = self.loc[self.loc[:, 'i'].isin(bridges_workers) & self.loc[:, 'j'].isin(bridges_firms), ['i', 'j', 'm']]
        # possible_articulation_obs = self.loc[pd.Series(map(tuple, self.loc[:, ['i', 'j']].to_numpy())).reindex_like(self, copy=False).isin(bridges), ['i', 'j', 'm']] # FIXME this doesn't work

        # Find articulation observations - an observation is an articulation observation if the firm-worker pair has only a single observation
        articulation_rows = possible_articulation_obs.index.to_numpy()[possible_articulation_obs.groupby(['i', 'j'], sort=(not (is_sorted and self.no_returns)))['m'].transform('size').to_numpy() == 1]

        return articulation_rows

    def fill_periods(self, fill_j=-1, fill_y=pd.NA, fill_m=pd.NA, is_sorted=False, copy=True):
        '''
        Return Pandas dataframe of long form data with missing periods filled in as unemployed. By default j is filled in as - 1, and y and m are filled in as pd.NA, but these values can be specified.

        Arguments:
            fill_j (value): value to fill in for missing j
            fill_y (value): value to fill in for missing y
            fill_m (value): value to fill in for missing m
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            fill_frame (Pandas DataFrame): Pandas DataFrame with missing periods filled in as unemployed
        '''
        m = self._col_included('m') # Check whether m column included
        fill_frame = pd.DataFrame(self, copy=copy)
        if not is_sorted:
            # Sort data by i, t
            fill_frame.sort_values(['i', 't'], inplace=True)
            fill_frame.reset_index(drop=True, inplace=True)
        # Find missing periods
        i_col = fill_frame.loc[:, 'i'].to_numpy()
        t_col = fill_frame.loc[:, 't'].to_numpy()
        i_prev = bpd.fast_shift(i_col, 1, fill_value=-2)
        t_prev = np.roll(t_col, 1)
        missing_periods = (i_col == i_prev) & (t_col != t_prev + 1)
        del i_prev
        if np.sum(missing_periods) > 0:
            # If have data to fill in
            fill_data = []
            for index in fill_frame.loc[missing_periods, :].index:
                row = fill_frame.iloc[index]
                # Only iterate over missing years
                for t in range(int(t_prev[index]) + 1, int(row.loc['t'])):
                    new_row = {'i': int(row.loc['i']), 'j': fill_j, 'y': fill_y, 't': t}
                    if m: # If m column included
                        new_row['m'] = fill_m # int(row['m'])
                    fill_data.append(new_row)
            fill_df = pd.concat([pd.DataFrame(fill_row, index=[i]) for i, fill_row in enumerate(fill_data)])
            fill_frame = pd.concat([fill_frame, fill_df])
            # Sort data by i, t
            fill_frame.sort_values(['i', 't'], inplace=True)
            fill_frame.reset_index(drop=True, inplace=True)

        return fill_frame

    def get_es_extended(self, periods_pre=3, periods_post=3, stable_pre=[], stable_post=[], include=['g', 'y'], transition_col='j', is_sorted=False, copy=True):
        '''
        Return Pandas dataframe of event study with periods_pre periods before the transition (the transition is defined by a switch in the transition column) and periods_post periods after the transition, where transition fulcrums are given by job moves, and the first post-period is given by the job move. Returned dataframe gives worker id, period of transition, income over all periods, and firm cluster over all periods. The function will run .cluster() if no g column exists.

        Arguments:
            periods_pre (int): number of periods before the transition
            periods_post (int): number of periods after the transition
            stable_pre (column name or list of column names): for each column, keep only workers who have constant values in that column before the transition
            stable_post (column name or list of column names): for each column, keep only workers who have constant values in that column after the transition
            include (column name or list of column names): columns to include data for all periods
            transition_col (str): column to use to define a transition
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            es_extended_frame or None (Pandas DataFrame or None): extended event study generated from long data if clustered; None if not clustered
        '''
        # Convert into lists
        include = bpd.to_list(include)
        stable_pre = bpd.to_list(stable_pre)
        stable_post = bpd.to_list(stable_post)

        # Get list of all columns (note that stable_pre and stable_post can have columns that are not in include)
        all_cols = include[:]
        for col in set(stable_pre + stable_post):
            if col not in all_cols:
                all_cols.append(col)

        # Check that columns exist
        for col in all_cols:
            if not self._col_included(col):
                return None

        # Create return frame
        es_extended_frame = pd.DataFrame(self, copy=copy)

        if not is_sorted:
            # Sort data by i and t
            es_extended_frame.sort_values(['i', 't'], inplace=True)

        # Generate how many periods each worker worked
        es_extended_frame.loc[:, 'one'] = 1
        es_extended_frame.loc[:, 'worker_total_periods'] = es_extended_frame.groupby('i', sort=False)['one'].transform('size')

        # Keep workers with enough periods (must have at least periods_pre + periods_post periods)
        es_extended_frame = es_extended_frame.loc[es_extended_frame.loc[:, 'worker_total_periods'].to_numpy() >= periods_pre + periods_post, :]
        es_extended_frame.reset_index(drop=True, inplace=True)

        # For each worker-period, generate (how many total years - 1) they have worked at that point (e.g. if a worker started in 2005, and had data each year, then 2008 would give 3, 2009 would give 4, etc.)
        es_extended_frame.loc[:, 'worker_periods_worked'] = es_extended_frame.groupby('i', sort=False)['one'].cumsum().to_numpy() - 1
        es_extended_frame.drop('one', axis=1, inplace=True)

        # Find periods where the worker transitioned, which can serve as fulcrums for the event study
        i_col = es_extended_frame.loc[:, 'i'].to_numpy()
        transition_col = es_extended_frame.loc[:, transition_col].to_numpy()
        i_prev = bpd.fast_shift(i_col, 1, fill_value=-2)
        transition_prev = np.roll(transition_col, 1)
        moved_firms = (i_col == i_prev) & (transition_col != transition_prev)
        es_extended_frame.loc[:, 'moved_firms'] = moved_firms
        del i_col, transition_col, i_prev, transition_prev, moved_firms

        # Compute valid moves - periods where the worker transitioned, and they also have periods_pre periods before the move, and periods_post periods after (and including) the move
        es_extended_frame.loc[:, 'valid_move'] = \
                                es_extended_frame.loc[:, 'moved_firms'] & \
                                (es_extended_frame.loc[:, 'worker_periods_worked'].to_numpy() >= periods_pre) & \
                                ((es_extended_frame.loc[:, 'worker_total_periods'].to_numpy() - es_extended_frame.loc[:, 'worker_periods_worked'].to_numpy()) >= periods_post)

        # Drop irrelevant columns
        es_extended_frame.drop(['worker_total_periods', 'worker_periods_worked', 'moved_firms'], axis=1, inplace=True)

        # Only keep workers who have a valid move
        es_extended_frame = es_extended_frame.loc[es_extended_frame.groupby('i', sort=False)['valid_move'].transform(max).to_numpy() > 0, :]

        ## Compute lags and leads
        # For column order
        column_order = [[] for _ in range(len(include))]
        for i, col in enumerate(all_cols):
            col_np = es_extended_frame.loc[:, col].to_numpy()
            # Compute lagged values
            for j in range(1, periods_pre + 1):
                es_extended_frame.loc[:, '{}_l{}'.format(col, j)] = bpd.fast_shift(col_np, j, fill_value=-2)
                if col in include:
                    column_order[i].insert(0, '{}_l{}'.format(col, j))
            # Compute lead values
            for j in range(periods_post): # No + 1 because base period has no shift (e.g. y becomes y_f1)
                if j > 0:
                    # No shift necessary for base period because already exists
                    es_extended_frame.loc[:, '{}_f{}'.format(col, j + 1)] = bpd.fast_shift(col_np, -j, fill_value=-2)
                if col in include:
                    column_order[i].append('{}_f{}'.format(col, j + 1))

        # Demarcate valid rows (all should start off True)
        valid_rows = ~pd.isna(es_extended_frame.loc[:, col])
        # Construct i and i_prev (these have changed from before)
        i_col = es_extended_frame.loc[:, 'i'].to_numpy()
        i_prev = bpd.fast_shift(i_col, 1, fill_value=-2)
        # Stable pre-trend
        for col in stable_pre:
            col_np = es_extended_frame.loc[:, col].to_numpy()
            for i in range(2, periods_pre + 1):
                # Shift 1 is baseline
                valid_rows = (valid_rows) & (np.roll(col_np, 1) == np.roll(col_np, i)) & (i_prev == bpd.fast_shift(i_col, i, fill_value=-3))

        # Stable post-trend
        for col in stable_post:
            col_np = es_extended_frame.loc[:, col].to_numpy()
            for i in range(1, periods_post):
                # Shift 0 is baseline
                valid_rows = (valid_rows) & (col_np == np.roll(col_np, -i)) & (i_col == bpd.fast_shift(i_col, -i, fill_value=-2))

        # Delete i_col, i_prev
        del i_col, i_prev

        # Update with pre- and/or post-trend
        es_extended_frame = es_extended_frame.loc[valid_rows, :]

        # Rename base period to have _f1 (e.g. y becomes y_f1)
        es_extended_frame.rename({col: col + '_f1' for col in include}, axis=1, inplace=True)

        # Keep rows with valid moves
        es_extended_frame = es_extended_frame.loc[es_extended_frame.loc[:, 'valid_move'].to_numpy() == 1, :]
        es_extended_frame.reset_index(drop=True, inplace=True)

        # Drop irrelevant columns
        es_extended_frame.drop('valid_move', axis=1, inplace=True)

        # Correct datatypes
        for i, col in enumerate(include):
            es_extended_frame.loc[:, column_order[i]] = es_extended_frame.loc[:, column_order[i]].astype(self.col_dtype_dict[col], copy=False)

        col_order = []
        for order in column_order:
            col_order += order

        # Reorder columns
        es_extended_frame = es_extended_frame.reindex(['i', 't'] + col_order, axis=1, copy=False)

        # Return es_extended_frame
        return es_extended_frame

    def plot_es_extended(self, periods_pre=2, periods_post=2, stable_pre=[], stable_post=[], include=['g', 'y'], transition_col='j', es_extended_plot_params=es_extended_plot_params(), is_sorted=False, copy=True):
        '''
        Generate event study plots. If data is not clustered, will plot all transitions in a single figure.

        Arguments:
            periods_pre (int): number of periods before the transition
            periods_post (int): number of periods after the transition
            stable_pre (column name or list of column names): for each column, keep only workers who have constant values in that column before the transition
            stable_post (column name or list of column names): for each column, keep only workers who have constant values in that column after the transition
            include (column name or list of column names): columns to include data for all periods
            transition_col (str): column to use to define a transition
            es_extended_plot_params (ParamsDict): dictionary of parameters for plotting. Run bpd.es_extended_plot_params().describe_all() for descriptions of all valid parameters.
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy
        '''
        # FIXME this method raises the following warnings:
        # ResourceWarning: unclosed event loop <_UnixSelectorEventLoop running=False closed=False debug=False> source=self)
        # ResourceWarning: Enable tracemalloc to get the object allocation traceback
        from matplotlib import pyplot as plt

        n_clusters = self.n_clusters()
        added_g = False
        if n_clusters is None:
            # If data not clustered
            added_g = True
            n_clusters = 1
            self.loc[:, 'g'] = 0
            self.col_dict['g'] = 'g'

        es = self.get_es_extended(periods_pre=periods_pre, periods_post=periods_post, stable_pre=stable_pre, stable_post=stable_post, include=include, transition_col=transition_col,is_sorted=is_sorted, copy=copy)

        # Want n_clusters x n_clusters subplots
        fig, axs = plt.subplots(nrows=n_clusters, ncols=n_clusters, sharex=es_extended_plot_params['sharex'], sharey=es_extended_plot_params['sharey'])
        # Create lists of the x values and y columns we want
        x_vals = []
        y_cols = []
        for i in range(1, periods_pre + 1):
            x_vals.insert(0, - i)
            y_cols.insert(0, 'y_l{}'.format(i))
        for i in range(1, periods_post + 1):
            x_vals.append(i)
            y_cols.append('y_f{}'.format(i))
        # Get y boundaries
        y_min = 1000
        y_max = -1000
        # Generate plots
        if n_clusters > 1:
            for i, row in enumerate(axs):
                for j, ax in enumerate(row):
                    # Keep if previous firm type is i and next firm type is j
                    es_plot = es.loc[(es.loc[:, 'g_l1'].to_numpy() == i) & (es.loc[:, 'g_f1'].to_numpy() == j), :]
                    y = es_plot.loc[:, y_cols].mean(axis=0)
                    yerr = es_plot.loc[:, y_cols].std(axis=0) / (len(es_plot) ** 0.5)
                    ax.errorbar(x_vals, y, yerr=yerr, ecolor='red', elinewidth=1, zorder=2)
                    ax.axvline(0, color='orange', zorder=1)
                    ax.set_title('{} to {} (n={})'.format(i + 1, j + 1, len(es_plot)), y=es_extended_plot_params['title_height'], fontdict={'fontsize': es_extended_plot_params['fontsize']})
                    ax.grid()
                    y_min = min(y_min, ax.get_ylim()[0])
                    y_max = max(y_max, ax.get_ylim()[1])
        else:
            # Plot everything
            y = es.loc[:, y_cols].mean(axis=0)
            yerr = es.loc[:, y_cols].std(axis=0) / (len(es) ** 0.5)
            axs.errorbar(x_vals, y, yerr=yerr, ecolor='red', elinewidth=1, zorder=2)
            axs.axvline(0, color='orange', zorder=1)
            axs.set_title('All Transitions (n={})'.format(len(es)), y=es_extended_plot_params['title_height'], fontdict={'fontsize': es_extended_plot_params['fontsize']})
            axs.grid()
            y_min = min(y_min, axs.get_ylim()[0])
            y_max = max(y_max, axs.get_ylim()[1])

        if added_g:
            # Drop g column
            self.drop('g', axis=1, inplace=True)
            self.col_dict['g'] = None

        # Plot
        plt.setp(axs, xticks=np.arange(-periods_pre, periods_post + 1), yticks=np.round(np.linspace(y_min, y_max, 4), es_extended_plot_params['yticks_round']))
        plt.tight_layout()
        plt.show()
