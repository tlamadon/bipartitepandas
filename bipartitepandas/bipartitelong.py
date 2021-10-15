'''
Class for a bipartite network in long form
'''
import pandas as pd
import bipartitepandas as bpd

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

        # self.logger.info('BipartiteLong object initialized')

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

    def get_collapsed_long(self, copy=True):
        '''
        Collapse long data by job spells (so each spell for a particular worker at a particular firm is one observation).

        Arguments:
            copy (bool): if False, avoid copy

        Returns:
            collapsed_frame (BipartiteLongCollapsed): BipartiteLongCollapsed object generated from long data collapsed by job spells
        '''
        # Generate m column (the function checks if it already exists)
        self.gen_m()

        # Convert to Pandas dataframe
        data = pd.DataFrame(self, copy=copy)
        # Sort data by i and t
        data = data.sort_values(['i', 't'])
        self.logger.info('copied data sorted by i and t')

        # Introduce lagged i and j
        data['i_l1'] = data['i'].shift(periods=1)
        data['j_l1'] = data['j'].shift(periods=1)
        self.logger.info('lagged i and j introduced')

        # Generate spell ids
        # Source: https://stackoverflow.com/questions/59778744/pandas-grouping-and-aggregating-consecutive-rows-with-same-value-in-column
        new_spell = (data['j'] != data['j_l1']) | (data['i'] != data['i_l1']) # Allow for i != i_l1 to ensure that consecutive workers at the same firm get counted as different spells
        data['spell_id'] = new_spell.cumsum()
        self.logger.info('spell ids generated')

        # Aggregate at the spell level
        spell = data.groupby(['spell_id'])
        # First, aggregate required columns
        data_spell = spell.agg(
            i=pd.NamedAgg(column='i', aggfunc='first'),
            j=pd.NamedAgg(column='j', aggfunc='first'),
            y=pd.NamedAgg(column='y', aggfunc='mean'),
            t1=pd.NamedAgg(column='t', aggfunc='min'),
            t2=pd.NamedAgg(column='t', aggfunc='max'),
            w=pd.NamedAgg(column='i', aggfunc='size')
        )
        # Next, aggregate optional columns
        all_cols = self._included_cols()
        for col in all_cols:
            if col in self.columns_opt:
                if self.col_dtype_dict[col] == 'int':
                    for subcol in bpd.to_list(self.reference_dict[col]):
                        data_spell[subcol] = spell[subcol].first()
                if self.col_dtype_dict[col] == 'float':
                    for subcol in bpd.to_list(self.reference_dict[col]):
                        data_spell[subcol] = spell[subcol].mean()

        # # Classify movers and stayers
        # if not self._col_included('m'):
        #     spell_count = data_spell.groupby(['i']).transform('count')['j'] # Choice of j arbitrary
        #     data_spell['m'] = (spell_count > 1).astype(int)
        collapsed_data = data_spell.reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(collapsed_data.columns, key=bpd.col_order)
        collapsed_data = collapsed_data[sorted_cols]

        self.logger.info('data aggregated at the spell level')

        collapsed_frame = bpd.BipartiteLongCollapsed(collapsed_data)
        collapsed_frame._set_attributes(self, no_dict=True)

        return collapsed_frame

    def fill_periods(self, fill_j=-1, fill_y=pd.NA, copy=True):
        '''
        Return Pandas dataframe of long form data with missing periods filled in as unemployed. By default j is filled in as - 1 and y is filled in as pd.NA, but these values can be specified.

        Arguments:
            fill_j (value): value to fill in for missing j
            fill_y (value): value to fill in for missing y
            copy (bool): if False, avoid copy

        Returns:
            fill_frame (Pandas DataFrame): Pandas DataFrame with missing periods filled in as unemployed
        '''
        import numpy as np
        m = self._col_included('m') # Check whether m column included
        fill_frame = pd.DataFrame(self, copy=copy).sort_values(['i', 't']).reset_index(drop=True) # Sort by i, t
        fill_frame['i_l1'] = fill_frame['i'].shift(periods=1) # Lagged value
        fill_frame['t_l1'] = fill_frame['t'].shift(periods=1) # Lagged value
        missing_periods = (fill_frame['i'] == fill_frame['i_l1']) & (fill_frame['t'] != fill_frame['t_l1'] + 1)
        if np.sum(missing_periods) > 0: # If have data to fill in
            fill_data = []
            for index in fill_frame[missing_periods].index:
                row = fill_frame.iloc[index]
                # Only iterate over missing years
                for t in range(int(row['t_l1']) + 1, int(row['t'])):
                    new_row = {'i': int(row['i']), 'j': fill_j, 'y': fill_y, 't': t}
                    if m: # If m column included
                        new_row['m'] = int(row['m'])
                    fill_data.append(new_row)
            fill_df = pd.concat([pd.DataFrame(fill_row, index=[i]) for i, fill_row in enumerate(fill_data)])
            fill_frame = pd.concat([fill_frame, fill_df]).sort_values(['i', 't']).reset_index(drop=True) # Sort by i, t
        fill_frame.drop(['i_l1', 't_l1'], axis=1, inplace=True)

        return fill_frame

    def get_es_extended(self, periods_pre=3, periods_post=3, stable_pre=[], stable_post=[], include=['g', 'y'], transition_col='j', copy=True):
        '''
        Return Pandas dataframe of event study with periods_pre periods before the transition (the transition is defined by a switch in the transition column) and periods_post periods after the transition, where transition fulcrums are given by job moves, and the first post-period is given by the job move. Returned dataframe gives worker id, period of transition, income over all periods, and firm cluster over all periods. The function will run .cluster() if no g column exists.

        Arguments:
            periods_pre (int): number of periods before the transition
            periods_post (int): number of periods after the transition
            stable_pre (column name or list of column names): for each column, keep only workers who have constant values in that column before the transition
            stable_post (column name or list of column names): for each column, keep only workers who have constant values in that column after the transition
            include (column name or list of column names): columns to include data for all periods
            transition_col (str): column to use to define a transition
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

        # Generate how many periods each worker worked
        es_extended_frame['one'] = 1
        es_extended_frame['worker_total_periods'] = es_extended_frame.groupby('i')['one'].transform(sum) # Must faster to use .transform(sum) than to use .transform(len)

        # Keep workers with enough periods (must have at least periods_pre + periods_post periods)
        es_extended_frame = es_extended_frame[es_extended_frame['worker_total_periods'] >= periods_pre + periods_post].reset_index(drop=True)

        # Sort by worker-period
        es_extended_frame.sort_values(['i', 't'], inplace=True)

        # For each worker-period, generate (how many total years - 1) they have worked at that point (e.g. if a worker started in 2005, and had data each year, then 2008 would give 3, 2009 would give 4, etc.)
        es_extended_frame['worker_periods_worked'] = es_extended_frame.groupby('i')['one'].cumsum() - 1
        es_extended_frame.drop('one', axis=1, inplace=True)

        # Find periods where the worker transitioned, which can serve as fulcrums for the event study
        es_extended_frame['moved_firms'] = ((es_extended_frame['i'] == es_extended_frame['i'].shift(periods=1)) & (es_extended_frame[transition_col] != es_extended_frame[transition_col].shift(periods=1))).astype(int)

        # Compute valid moves - periods where the worker transitioned, and they also have periods_pre periods before the move, and periods_post periods after (and including) the move
        es_extended_frame['valid_move'] = \
                                es_extended_frame['moved_firms'] & \
                                (es_extended_frame['worker_periods_worked'] >= periods_pre) & \
                                ((es_extended_frame['worker_total_periods'] - es_extended_frame['worker_periods_worked']) >= periods_post)

        # Drop irrelevant columns
        es_extended_frame.drop(['worker_total_periods', 'worker_periods_worked', 'moved_firms'], axis=1, inplace=True)

        # Only keep workers who have a valid move
        es_extended_frame = es_extended_frame[es_extended_frame.groupby('i')['valid_move'].transform(max) > 0]

        # Compute lags and leads
        column_order = [[] for _ in range(len(include))] # For column order
        for i, col in enumerate(all_cols):
            # Compute lagged values
            for j in range(1, periods_pre + 1):
                es_extended_frame['{}_l{}'.format(col, j)] = es_extended_frame[col].shift(periods=j)
                if col in include:
                    column_order[i].insert(0, '{}_l{}'.format(col, j))
            # Compute lead values
            for j in range(periods_post): # No + 1 because base period has no shift (e.g. y becomes y_f1)
                if j > 0: # No shift necessary for base period because already exists
                    es_extended_frame['{}_f{}'.format(col, j + 1)] = es_extended_frame[col].shift(periods=-j)
                if col in include:
                    column_order[i].append('{}_f{}'.format(col, j + 1))

        valid_rows = ~pd.isna(es_extended_frame[col]) # Demarcate valid rows (all should start off True)
        # Stable pre-trend
        for col in stable_pre:
            for i in range(2, periods_pre + 1): # Shift 1 is baseline
                valid_rows = (valid_rows) & (es_extended_frame[col].shift(periods=1) == es_extended_frame[col].shift(periods=i))

        # Stable post-trend
        for col in stable_post:
            for i in range(1, periods_post): # Shift 0 is baseline
                valid_rows = (valid_rows) & (es_extended_frame[col] == es_extended_frame[col].shift(periods=-i))

        # Update with pre- and/or post-trend
        es_extended_frame = es_extended_frame[valid_rows]

        # Rename base period to have _f1 (e.g. y becomes y_f1)
        es_extended_frame.rename({col: col + '_f1' for col in include}, axis=1, inplace=True)

        # Keep rows with valid moves
        es_extended_frame = es_extended_frame[es_extended_frame['valid_move'] == 1].reset_index(drop=True)

        # Drop irrelevant columns
        es_extended_frame.drop('valid_move', axis=1, inplace=True)

        # Correct datatypes
        for i, col in enumerate(include):
            es_extended_frame[column_order[i]] = es_extended_frame[column_order[i]].astype(self.col_dtype_dict[col])

        col_order = []
        for order in column_order:
            col_order += order

        # Reorder columns
        es_extended_frame = es_extended_frame[['i', 't'] + col_order]

        # Return es_extended_frame
        return es_extended_frame

    def plot_es_extended(self, periods_pre=2, periods_post=2, stable_pre=[], stable_post=[], include=['g', 'y'], transition_col='j', user_graph={}, user_clean={}):
        '''
        Generate event study plots.
        Arguments:
            periods_pre (int): number of periods before the transition
            periods_post (int): number of periods after the transition
            stable_pre (column name or list of column names): for each column, keep only workers who have constant values in that column before the transition
            stable_post (column name or list of column names): for each column, keep only workers who have constant values in that column after the transition
            include (column name or list of column names): columns to include data for all periods
            transition_col (str): column to use to define a transition
            user_graph (dict): dictionary of parameters for graphing
                Dictionary parameters:
                    title_height (float, default=-0.45): location of titles for subfigures
                    fontsize (float, default=9): font size of titles for subfigures
            user_clean (dict): dictionary of parameters for cleaning
                Dictionary parameters:
                    connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations
                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids
                    data_validity (bool, default=True): if True, run data validity checks; much faster if set to False
                    copy (bool, default=False): if False, avoid copy
        '''
        import numpy as np
        from matplotlib import pyplot as plt

        # Default parameter dictionaries
        default_graph = {
            'title_height': -0.45,
            'fontsize': 9
        }
        graph_params = bpd.update_dict(default_graph, user_graph)

        es = self.get_es_extended(periods_pre=periods_pre, periods_post=periods_post, stable_pre=stable_pre, stable_post=stable_post, include=include, transition_col=transition_col)
        n_clusters = self.n_clusters()
        # Want n_clusters x n_clusters subplots
        fig, axs = plt.subplots(nrows=n_clusters, ncols=n_clusters)
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
        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                # Keep if previous firm type is i and next firm type is j
                es_plot = es[(es['g_l1'] == i) & (es['g_f1'] == j)]
                y = es_plot[y_cols].mean(axis=0)
                yerr = es_plot[y_cols].std(axis=0) / (len(es_plot) ** 0.5)
                ax.errorbar(x_vals, y, yerr=yerr, ecolor='red', elinewidth=1, zorder=2)
                ax.axvline(0, color='orange', zorder=1)
                ax.set_title('{} to {} (n={})'.format(i, j, len(es_plot)), y=graph_params['title_height'], fontdict={'fontsize': graph_params['fontsize']})
                ax.grid()
                y_min = min(y_min, ax.get_ylim()[0])
                y_max = max(y_max, ax.get_ylim()[1])
        plt.setp(axs, xticks=np.arange(-periods_pre, periods_post + 1), yticks=np.round(np.linspace(y_min, y_max, 4), 1))
        plt.tight_layout()
        plt.show()
