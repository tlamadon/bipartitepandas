'''
Base class for bipartite networks in event study or collapsed event study format.
'''
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import bipartitepandas as bpd

class BipartiteEventStudyBase(bpd.BipartiteBase):
    '''
    Base class for BipartiteEventStudy and BipartiteEventStudyCollapsed, where BipartiteEventStudy and BipartiteEventStudyCollapsed give a bipartite network of firms and workers in event study and collapsed event study form, respectively. Contains generalized methods. Inherits from BipartiteBase.

    Arguments:
        *args: arguments for BipartiteBase
        col_reference_dict (dict or None): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}; None is equivalent to {}
        **kwargs: keyword arguments for BipartiteBase
    '''

    def __init__(self, *args, col_reference_dict=None, **kwargs):
        # Update parameters to be lists/dictionaries instead of None (source: https://stackoverflow.com/a/54781084/17333120)
        if col_reference_dict is None:
            col_reference_dict = {}
        col_reference_dict = bpd.util.update_dict({'j': ['j1', 'j2'], 'y': ['y1', 'y2'], 'g': ['g1', 'g2'], 'w': ['w1', 'w2']}, col_reference_dict)
        # Initialize DataFrame
        super().__init__(*args, col_reference_dict=col_reference_dict, **kwargs)

        # self.log('BipartiteEventStudyBase object initialized', level='info')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.

        Returns:
            (class): BipartiteEventStudyBase class
        '''
        return BipartiteEventStudyBase

    def gen_m(self, force=False, copy=True):
        '''
        Generate m column for data (m == 0 if stayer, m == 1 or 2 if mover).

        Arguments:
            force (bool): if True, reset 'm' column even if it exists
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe with m column
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if not frame._col_included('m') or force:
            frame.loc[:, 'm'] = (frame.loc[:, 'j1'].to_numpy() != frame.loc[:, 'j2'].to_numpy()).astype(int, copy=False)
            # frame.loc[:, 'm'] = frame.groupby('i')['m'].transform('max')

            # Sort columns
            frame = frame.sort_cols(copy=False)

        else:
            self.log("'m' column already included. Returning unaltered frame.", level='info')

        return frame

    def clean(self, params=None):
        '''
        Clean data to make sure there are no NaN or duplicate observations, observations where workers leave a firm then return to it are removed, firms are connected by movers, and categorical ids are contiguous.

        Arguments:
            params (ParamsDict or None): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.clean_params().

        Returns:
            (BipartiteEventStudyBase): dataframe with cleaned data
        '''
        if params is None:
            params = bpd.clean_params()

        self.log('beginning BipartiteEventStudyBase data cleaning', level='info')

        verbose = params['verbose']

        # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
        no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

        # We copy when we generate 'm' (if the user specified to make a copy), then sort during the conversion to long, so we don't need to do these again when we clean the data after we convert it to long format
        params_copy = params.copy()
        params_copy.update({'is_sorted': True, 'copy': False})

        # Generate 'm' column - this is necessary for the next steps (note: 'm' will get updated in the following steps as it changes)
        self.log("generating 'm' column", level='info')
        if verbose:
            tqdm.write('checking required columns and datatypes')
        frame = self.gen_m(force=True, copy=params['copy'])

        # Clean long data, then convert back to event study (note: we use is_clean=False because duplicates mean that we should fully unstack all observations, to see which are duplicates and which are legitimate - setting is_clean=True would arbitrarily decide which rows are already correct)
        self.log('converting data to long format', level='info')
        if verbose:
            tqdm.write('converting data to long format')
        frame = frame.to_long(is_clean=False, drop_no_split_columns=False, is_sorted=params['is_sorted'], copy=False)

        frame.drop_duplicates(inplace=True)

        frame = frame.clean(params_copy)

        self.log('converting data back to event study format', level='info')
        if verbose:
            tqdm.write('converting data back to event study format')
        frame = frame.to_eventstudy(is_sorted=True, copy=False)

        # Update col_long_es_dict for columns that aren't supposed to convert to long
        for col in no_split_cols:
            frame.col_long_es_dict[col] = None

        self.log('BipartiteEventStudyBase data cleaning complete', level='info')

        return frame

    def _get_unstack_rows(self, worker_m=None, is_sorted=False, copy=True):
        '''
        Get mask of rows where the second observation isn't included in the first observation for any rows (e.g., the last observation for a mover is given only as an j2, never as j1). More details: if a worker's last observation is a move OR if a worker switches from a move to a stay between observations, we need to append the second observation from the move. We can use the fact that the time changes between observations to indicate that we need to unstack, because event studies are supposed to go A -> B, B -> C, so if it goes A -> B, C -> D that means B should be unstacked.

        Arguments:
            worker_m (NumPy Array or None): if a NumPy Array, this gives the worker_m column; if None, generate this column
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (NumPy Array): mask of rows to unstack
        '''
        if not is_sorted:
            raise NotImplementedError('._get_unstack_rows() requires `is_sorted` == True, but it is set to False.')

        # Sort and copy
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)

        # Get i for this and next period
        i_col = frame.loc[:, 'i'].to_numpy()
        i_next = bpd.util.fast_shift(i_col, -1, fill_value=-2)
        # Get m
        if worker_m is None:
            worker_m = frame.get_worker_m(is_sorted=True) # m_col = (self.loc[:, 'm'].to_numpy() > 0)
        # # Get t for this and next period
        # t_cols = bpd.util.to_list(self.col_reference_dict['t'])
        # halfway = len(t_cols) // 2
        # t1 = t_cols[0]
        # t2 = t_cols[halfway]
        # t1_next = np.roll(self.loc[:, t1].to_numpy(), -1)
        # t2_col = self.loc[:, t2].to_numpy()
        # # Check if t changed
        # t_change = (i_col == i_next) & (t1_next != t2_col)
        # Check if i changed
        # Source: https://stackoverflow.com/a/47115520/17333120
        i_last = (i_col != i_next)

        return worker_m & i_last # m_col & (t_change | i_last)

    def diagnostic(self):
        '''
        Run diagnostic and print diagnostic report.
        '''
        super().diagnostic()

        if self._col_included('m'):
            ret_str = '----- Event Study Diagnostic -----\n'
            stayers = self.loc[self.loc[:, 'm'].to_numpy() == 0, :]
            movers = self.loc[self.loc[:, 'm'].to_numpy() > 0, :]

            ##### Firms #####
            firms_stayers = (stayers.loc[:, 'j1'].to_numpy() != stayers.loc[:, 'j2'].to_numpy()).sum()
            firms_movers = (movers.loc[:, 'j1'].to_numpy() == movers.loc[:, 'j2'].to_numpy()).sum()

            ret_str += f'm==0 with different firms (should be 0): {firms_stayers}\n'
            ret_str += f'm>0 with same firm (should be 0): {firms_movers}\n'

            ##### Income #####
            income_stayers = (stayers.loc[:, 'y1'].to_numpy() != stayers.loc[:, 'y2'].to_numpy()).sum()

            ret_str += f'm==0 with different income (should be 0): {income_stayers}'

            print(ret_str)

    def _drop_i_t_duplicates(self, how='max', is_sorted=False, copy=True):
        '''
        Keep only the highest paying job for i-t (worker-year) duplicates.

        Arguments:
            how (str or function): if 'max', keep max paying job; otherwise, take `how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `how` can take any input valid for a Pandas transform.
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe that keeps only the highest paying job for i-t (worker-year) duplicates. If no t column(s), returns frame with no changes
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if frame._col_included('t'):
            ## Convert to long
            # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
            no_split_cols = [col for col, long_es_split in frame.col_long_es_dict.items() if long_es_split is None]

            # Drop i-t duplicates for long data, then convert back to event study (note: we use is_clean=False because duplicates mean that we should fully unstack all observations, to see which are duplicates and which are legitimate - setting is_clean=True would arbitrarily decide which rows are already correct)
            frame = frame.to_long(is_clean=False, drop_no_split_columns=False, is_sorted=is_sorted, copy=False)

            frame.drop_duplicates(inplace=True)

            frame = frame._drop_i_t_duplicates(how, is_sorted=True, copy=False).to_eventstudy(is_sorted=True, copy=False)

            # Update col_long_es_dict for columns that aren't supposed to convert to long
            for col in no_split_cols:
                frame.col_long_es_dict[col] = None

            # Data now has unique i-t observations
            frame.i_t_unique = True
        else:
            frame.i_t_unique = None

        return frame

    def get_cs(self, copy=True):
        '''
        Return (collapsed) event study data reformatted into cross section data.

        Arguments:
            copy (bool): if False, avoid copy

        Returns:
            (Pandas DataFrame): cross section data
        '''
        sdata = pd.DataFrame(self.loc[self.loc[:, 'm'].to_numpy() == 0, :], copy=copy)
        jdata = pd.DataFrame(self.loc[self.loc[:, 'm'].to_numpy() > 0, :], copy=copy)

        # Columns used for constructing cross section
        cs_cols = self._included_cols(subcols=True)

        # Dictionary to swap names for cs=0 (these rows contain period-2 data for movers, so must swap columns for all relevant information to be contained in the same column (e.g. must move y2 into y1, otherwise bottom rows are just duplicates))
        rename_dict = {}
        for col in self._included_cols():
            subcols = bpd.util.to_list(self.col_reference_dict[col])
            n_subcols = len(subcols)
            # If even number of subcols, then is formatted as 'x1', 'x2', etc., so must swap to be 'x2', 'x1', etc.
            if n_subcols % 2 == 0:
                halfway = n_subcols // 2
                for i in range(halfway):
                    rename_dict[subcols[i]] = subcols[halfway + i]
                    rename_dict[subcols[halfway + i]] = subcols[i]

        # Combine the 2 data-sets
        data_cs = pd.concat([
            sdata.loc[:, cs_cols].assign(cs=1),
            jdata.loc[:, cs_cols].assign(cs=1),
            jdata.loc[:, cs_cols].rename(rename_dict, axis=1).assign(cs=0)
        ], ignore_index=True)

        # Sort columns
        sorted_cols = bpd.util._sort_cols(data_cs.columns)
        data_cs = data_cs.reindex(sorted_cols, axis=1, copy=False)

        self.log('mover and stayer event study datasets combined into cross section', level='info')

        return data_cs

    def to_long(self, is_clean=True, drop_no_split_columns=True, is_sorted=False, copy=True):
        '''
        Return (collapsed) event study data reformatted into (collapsed) long form.

        Arguments:
            is_clean (bool): if True, data is already clean (this ensures that observations that are in two consecutive event studies appear only once, e.g. the event study A -> B, B -> C turns into A -> B -> C; otherwise, it will become A -> B -> B -> C). Set to False if duplicates will be handled manually.
            drop_no_split_columns (bool): if True, columns marked by self.col_long_es_dict as None (i.e. they should be dropped) will not be dropped
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteLongBase): long format dataframe generated from event study data
        '''
        if not self._col_included('t'):
            raise NotImplementedError("Cannot convert from event study to long format without a time column. To bypass this, if you know your data is ordered by time but do not have time data, it is recommended to construct an artificial time column by calling .construct_artificial_time(copy=False).")

        # Sort and copy
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)

        # Keep track of user-added columns
        user_added_cols = {}
        default_cols = frame.columns_req + frame.columns_opt

        # Dictionary to swap names (necessary for last row of data, where period-2 observations are not located in subsequent period-1 column (as it doesn't exist), so must append the last row with swapped column names)
        rename_dict_1 = {}
        # Dictionary to reformat names into (collapsed) long form
        rename_dict_2 = {}
        # For casting column types
        astype_dict = {}
        # Columns to drop
        drops = []
        for col in frame._included_cols():
            if (frame.col_long_es_dict[col] is None) and drop_no_split_columns:
                # If None and data is clean, drop this column
                drops += bpd.util.to_list(frame.col_reference_dict[col])
            elif frame.col_long_es_dict[col]:
                # If column has been split
                subcols = bpd.util.to_list(frame.col_reference_dict[col])
                halfway = len(subcols) // 2
                if (col != 'i') and (len(subcols) % 2 == 1):
                    raise ValueError(f'{col!r} is listed as being split, but has an odd number of subcolumns. If this is a custom column, please make sure when adding it to the dataframe to specify long_es_split=False, to ensure it is not marked as being split, or long_es_split=None, to indicate the column should be dropped when converting between long and event study formats.')
                for i in range(halfway):
                    rename_dict_1[subcols[i]] = subcols[halfway + i]
                    rename_dict_1[subcols[halfway + i]] = subcols[i]
                    # Get column number, e.g. j1 will give 1
                    subcol_number = subcols[i].strip(col)
                    # Get rid of first number, e.g. j12 to j2 (note there is no indexing issue even if subcol_number has only one digit)
                    rename_dict_2[subcols[i]] = col + subcol_number[1:]
                    if frame.col_dtype_dict[col] in ['int', 'categorical']:
                        astype_dict[rename_dict_2[subcols[i]]] = int

                    if col not in default_cols:
                        # User-added columns
                        if col in user_added_cols.keys():
                            user_added_cols[col].append(col + subcol_number[1:])
                        else:
                            user_added_cols[col] = [col + subcol_number[1:]]

                    drops.append(subcols[halfway + i])

            else:
                # If column has not been split
                if frame.col_dtype_dict[col] in ['int', 'categorical']:
                    # Check correct type for other columns
                    for subcol in bpd.util.to_list(frame.col_reference_dict[col]):
                        astype_dict[subcol] = int
                if col not in default_cols:
                    # User-added columns
                    user_added_cols[col] = frame.col_reference_dict[col]

        if is_clean:
            # Find rows to unstack
            unstack_df = pd.DataFrame(frame.loc[frame._get_unstack_rows(is_sorted=True, copy=False), :])
        else:
            # If data isn't clean, just unstack all moves and deal with duplicates later
            unstack_df = pd.DataFrame(frame.loc[frame.get_worker_m(is_sorted=True), :])
        unstack_df.rename(rename_dict_1, axis=1, inplace=True)

        try:
            data_long = pd.concat([pd.DataFrame(frame), unstack_df], ignore_index=True)
            data_long.drop(drops, axis=1, inplace=True)
            data_long.rename(rename_dict_2, axis=1, inplace=True)
            data_long = data_long.astype(astype_dict, copy=False)
        except ValueError:
            # If nan values, use Int64
            for col in astype_dict.keys():
                astype_dict[col] = 'Int64'
            data_long = pd.concat([pd.DataFrame(frame), unstack_df], ignore_index=True)
            data_long.drop(drops, axis=1, inplace=True)
            data_long.rename(rename_dict_2, axis=1, inplace=True)
            data_long = data_long.astype(astype_dict, copy=False)

        ## Final steps
        # Sort columns
        sorted_cols = bpd.util._sort_cols(data_long.columns)
        data_long = data_long.reindex(sorted_cols, axis=1, copy=False)

        # Construct BipartiteLongBase dataframe
        long_frame = frame._constructor_long(data_long, col_reference_dict=user_added_cols, log=frame._log_on_indicator)
        long_frame._set_attributes(frame, no_dict=True)

        if not drop_no_split_columns:
            # If shouldn't drop None columns, set None columns to have split of False (this is because we don't want the column to drop during data cleaning)
            for col, long_es_split in frame.col_long_es_dict.items():
                if long_es_split is None:
                    long_frame.col_long_es_dict[col] = False
        else:
            # If should drop None columns
            for col, long_es_split in frame.col_long_es_dict.items():
                # Remove dropped columns from attribute dictionaries
                if long_es_split is None:
                    # If column should be dropped during conversion to event study format
                    del long_frame.col_dtype_dict[col]
                    del long_frame.col_collapse_dict[col]
                    del long_frame.col_long_es_dict[col]
                    if col in frame.columns_contig.keys():
                        # If column is categorical
                        del long_frame.columns_contig[col]
                        if long_frame.id_reference_dict:
                            # If linking contiguous ids to original ids
                            del long_frame.id_reference_dict[col]

        # Sort rows by i (and t, if included)
        long_frame = long_frame.sort_rows(is_sorted=False, copy=False)

        # Reset index
        long_frame.reset_index(drop=True, inplace=True)

        # Generate 'm' column
        long_frame = long_frame.gen_m(force=True, copy=False)

        return long_frame

    def _drop_returns(self, how=False, is_sorted=False, reset_index=False, copy=True):
        '''
        Drop observations where workers leave a firm then return to it.

        Arguments:
            how (str): if 'returns', drop observations where workers leave a firm then return to it; if 'returners', drop workers who ever leave then return to a firm; if 'keep_first_returns', keep first spell where a worker leaves a firm then returns to it; if 'keep_last_returns', keep last spell where a worker leaves a firm then returns to it; if False, keep all observations
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): not used for event study format
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe that drops observations where workers leave a firm then return to it
        '''
        # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
        no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

        # Drop returns
        frame = self.to_long(drop_no_split_columns=False, is_sorted=is_sorted, copy=copy)._drop_returns(how=how, is_sorted=True, reset_index=False, copy=False).to_eventstudy(is_sorted=True, copy=False)

        # Update col_long_es_dict for columns that aren't supposed to convert to long
        for col in no_split_cols:
            frame.col_long_es_dict[col] = None

        return frame

    def _prep_cluster(self, stayers_movers=None, t=None, weighted=True, is_sorted=False, copy=False):
        '''
        Prepare data for clustering.

        Arguments:
            stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers; if 'stays', clusters on only stays; if 'moves', clusters on only moves
            t (int or list of int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data); if list of int, gives periods in data to consider (only valid for non-collapsed data)
            weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy
        Returns:
            data (Pandas DataFrame): data prepared for clustering
            weights (NumPy Array or None): if weighted=True, gives NumPy array of firm weights for clustering; otherwise, is None
            jids (NumPy Array): firm ids of firms in subset of data used to cluster
        '''
        return self.to_long(is_sorted=is_sorted, copy=copy)._prep_cluster(stayers_movers=stayers_movers, t=t, weighted=weighted, is_sorted=True, copy=False)

    def _leave_out_observation_spell_match(self, cc_list, max_j, leave_out_group, component_size_variable='firms', drop_returns_to_stays=False, frame_largest_cc=None, is_sorted=False, copy=True, first_loop=True):
        '''
        Extract largest leave-one-(observation/spell/match)-out connected component.

        Arguments:
            cc_list (list of lists): each entry is a connected component
            max_j (int): maximum j in graph
            leave_out_group (str): which type of leave-one-out connected component to compute (options are 'observation', 'spell', or 'match')
            component_size_variable (str): how to determine largest leave-one-(observation/spell/match)-out connected component. Options are 'len'/'length' (length of frames), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), 'movers' (number of unique movers), 'firms_plus_workers' (number of unique firms + number of unique workers), 'firms_plus_stayers' (number of unique firms + number of unique stayers), 'firms_plus_movers' (number of unique firms + number of unique movers), 'len_stayers'/'length_stayers' (number of stayer observations), 'len_movers'/'length_movers' (number of mover observations), 'stays' (number of stay observations), and 'moves' (number of move observations).
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-(observation/spell/match)-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            frame_largest_cc (BipartiteLongBase): dataframe of baseline largest leave-one-(observation/spell/match)-out connected component
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy
            first_loop (bool): if True, this is the first loop of the method

        Returns:
            (BipartiteEventStudyBase): dataframe of largest leave-one-(observation/spell/match)-out connected component
        '''
        # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
        no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

        # Compute leave-one-(observation/spell/match)-out connected components
        frame = self.to_long(drop_no_split_columns=False, is_sorted=is_sorted, copy=copy)._leave_out_observation_spell_match(cc_list=cc_list, max_j=max_j, leave_out_group=leave_out_group, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays, frame_largest_cc=frame_largest_cc, is_sorted=True, copy=False, first_loop=first_loop).to_eventstudy(is_sorted=True, copy=False)

        # Update col_long_es_dict for columns that aren't supposed to convert to long
        for col in no_split_cols:
            frame.col_long_es_dict[col] = None

        return frame

    def _leave_out_worker(self, cc_list, max_j, component_size_variable='firms', drop_returns_to_stays=False, frame_largest_cc=None, is_sorted=False, copy=True, first_loop=True):
        '''
        Extract largest leave-one-worker-out connected component.

        Arguments:
            cc_list (list of lists): each entry is a connected component
            max_j (int): maximum j in graph
            component_size_variable (str): how to determine largest leave-one-worker-out connected component. Options are 'len'/'length' (length of frame), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), and 'movers' (number of unique movers)
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            frame_largest_cc (BipartiteLongBase): dataframe of baseline largest leave-one-worker-out connected component
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy
            first_loop (bool): if True, this is the first loop of the method

        Returns:
            (BipartiteEventStudyBase): dataframe of largest leave-one-worker-out connected component
        '''
        # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
        no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

        # Compute leave-one-worker-out connected components
        frame = self.to_long(drop_no_split_columns=False, is_sorted=is_sorted, copy=copy)._leave_out_worker(cc_list=cc_list, max_j=max_j, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays, frame_largest_cc=frame_largest_cc, is_sorted=True, copy=False, first_loop=first_loop).to_eventstudy(is_sorted=True, copy=False)

        # Update col_long_es_dict for columns that aren't supposed to convert to long
        for col in no_split_cols:
            frame.col_long_es_dict[col] = None

        return frame

    def _construct_firm_linkages(self, is_sorted=False, copy=True):
        '''
        Construct numpy array linking firms by movers, for use with connected components.

        Arguments:
            is_sorted (bool): not used for ._construct_firm_linkages()
            copy (bool): not used for ._construct_firm_linkages()

        Returns:
            (NumPy Array): firm linkages
            (int): maximum firm id
        '''
        return self.loc[(self.loc[:, 'm'].to_numpy() > 0), ['j1', 'j2']].to_numpy()

    def _construct_firm_double_linkages(self, is_sorted=False, copy=True):
        '''
        Construct numpy array linking firms by movers, for use with leave-one-firm-out components.

        Arguments:
            is_sorted (bool): not used for ._construct_firm_double_linkages()
            copy (bool): not used for ._construct_firm_double_linkages()

        Returns:
            (NumPy Array): firm linkages
            (int): maximum firm id
        '''
        move_rows = (self.loc[:, 'm'].to_numpy() > 0)
        base_linkages = self.loc[move_rows, ['j1', 'j2']].to_numpy()
        i_col = self.loc[move_rows, 'i'].to_numpy()
        i_next = bpd.util.fast_shift(i_col, -1, fill_value=-2)
        j2_next = np.roll(base_linkages[:, 1], -1)
        valid_i = (i_col == i_next)
        secondary_linkages = np.stack([base_linkages[valid_i, 1], j2_next[valid_i]], axis=1)
        linkages = np.concatenate([base_linkages, secondary_linkages], axis=0)
        max_j = np.max(linkages)

        return linkages, max_j

    def _construct_firm_worker_linkages(self, is_sorted=False, copy=True):
        '''
        Construct numpy array linking firms to worker ids, for use with leave-one-observation-out components.

        Arguments:
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (NumPy Array): firm-worker linkages
            (int): maximum firm id
        '''
        if not is_sorted:
            raise NotImplementedError('._construct_firm_worker_linkages() requires `is_sorted` == True, but it is set to False.')

        # Sort and copy
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)

        worker_m = frame.get_worker_m(is_sorted=True)
        base_linkages = frame.loc[worker_m, ['i', 'j1']].to_numpy()
        secondary_linkages = frame.loc[frame._get_unstack_rows(worker_m=worker_m, is_sorted=True, copy=False), ['i', 'j2']].to_numpy()
        linkages = np.concatenate([base_linkages, secondary_linkages], axis=0)
        max_j = np.max(linkages)

        return linkages, max_j

    def keep_ids(self, id_col, keep_ids_list, drop_returns_to_stays=False, is_sorted=False, reset_index=False, copy=True):
        '''
        Only keep ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            keep_ids_list (list): ids to keep
            drop_returns_to_stays (bool): used only if id_col is 'j' or 'g' and using BipartiteEventStudyCollapsed format. If True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer).
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): not used for event study format
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe with ids in the given set
        '''
        keep_ids_list = set(keep_ids_list)
        if len(keep_ids_list) == self.n_unique_ids(id_col):
            # If keeping everything
            if copy:
                return self.copy()
            return self

        # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
        no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

        # Keep ids
        frame = self.to_long(drop_no_split_columns=False, is_sorted=is_sorted, copy=copy).keep_ids(id_col=id_col, keep_ids_list=keep_ids_list, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False).to_eventstudy(is_sorted=True, copy=False)

        # Update col_long_es_dict for columns that aren't supposed to convert to long
        for col in no_split_cols:
            frame.col_long_es_dict[col] = None

        return frame

    def drop_ids(self, id_col, drop_ids_list, drop_returns_to_stays=False, is_sorted=False, reset_index=False, copy=True):
        '''
        Drop ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            drop_ids_list (list): ids to drop
            drop_returns_to_stays (bool): used only if id_col is 'j' or 'g' and using BipartiteEventStudyCollapsed format. If True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer).
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): not used for event study format
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe with ids outside the given set
        '''
        drop_ids_list = set(drop_ids_list)
        if len(drop_ids_list) == 0:
            # If nothing input
            if copy:
                return self.copy()
            return self

        # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
        no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

        # Drop ids
        frame = self.to_long(drop_no_split_columns=False, is_sorted=is_sorted, copy=copy).drop_ids(id_col=id_col, drop_ids_list=drop_ids_list, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False).to_eventstudy(is_sorted=True, copy=False)

        # Update col_long_es_dict for columns that aren't supposed to convert to long
        for col in no_split_cols:
            frame.col_long_es_dict[col] = None

        return frame

    def keep_rows(self, rows, drop_returns_to_stays=False, is_sorted=False, reset_index=False, copy=True):
        '''
        Only keep particular rows.

        Arguments:
            rows (list): rows to keep
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): not used for event study format
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe with given rows
        '''
        rows = set(rows)
        if len(rows) == len(self):
            # If keeping everything
            if copy:
                return self.copy()
            return self

        rows_list = sorted(list(rows_list))

        frame = self.iloc[rows_list]

        if isinstance(frame, bpd.BipartiteEventStudyCollapsed) and (not frame.no_returns):
            ## If BipartiteEventStudyCollapsed and there are returns, we have to recollapse
            # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
            no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

            # Recollapse
            frame = frame.to_long(drop_no_split_columns=False, is_sorted=is_sorted, copy=copy).recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, copy=False).to_eventstudy(is_sorted=True, copy=False)

            # We don't need to copy again
            copy = False

            # Update col_long_es_dict for columns that aren't supposed to convert to long
            for col in no_split_cols:
                frame.col_long_es_dict[col] = None

        if copy:
            return frame.copy()

        return frame

    def min_obs_firms(self, threshold=2, is_sorted=False, copy=True):
        '''
        List firms with at least `threshold` many observations.

        Arguments:
            threshold (int): minimum number of observations required to keep a firm
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (NumPy Array): firms with sufficiently many observations
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        # Sort and copy
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)

        # We consider j1 for all rows, but j2 only for unstack rows
        j_obs = frame.loc[:, 'j1'].value_counts(sort=False).to_dict()
        j2_obs = frame.loc[frame._get_unstack_rows(is_sorted=True, copy=False), 'j2'].value_counts(sort=False).to_dict()
        for j, n_obs in j2_obs.items():
            try:
                # If firm j in j1, add the observations in j2
                j_obs[j] += n_obs
            except KeyError:
                # If firm j not in j1, set observations to be j2
                j_obs[j] = n_obs

        valid_firms = []
        for j, n_obs in j_obs.items():
            if n_obs >= threshold:
                valid_firms.append(j)

        return np.array(valid_firms)

    def min_obs_frame(self, threshold=2, drop_returns_to_stays=False, is_sorted=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many observations.

        Arguments:
            threshold (int): minimum number of observations required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe of firms with sufficiently many observations
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
        no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

        # Compute min_obs_frame
        frame = self.to_long(drop_no_split_columns=False, is_sorted=is_sorted, copy=copy).min_obs_frame(threshold=threshold, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, copy=False).to_eventstudy(is_sorted=True, copy=False)

        # Update col_long_es_dict for columns that aren't supposed to convert to long
        for col in no_split_cols:
            frame.col_long_es_dict[col] = None

        return frame

    def min_workers_firms(self, threshold=2, is_sorted=False, copy=True):
        '''
        List firms with at least `threshold` many workers.

        Arguments:
            threshold (int): minimum number of workers required to keep a firm
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (NumPy Array): firms with sufficiently many workers
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        # Sort and copy
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)

        # We consider j1 for all rows, but j2 only for unstack rows
        j_i_ids = frame.groupby('j1')['i'].unique().apply(list).to_dict()
        j2_i_ids = frame.loc[frame._get_unstack_rows(is_sorted=True, copy=False), :].groupby('j2')['i'].unique().apply(list).to_dict()
        for j, i_ids in j2_i_ids.items():
            try:
                # If firm j in j1, add the worker ids in j2
                j_i_ids[j] += i_ids
            except KeyError:
                # If firm j not in j1, set worker ids to be j2
                j_i_ids[j] = i_ids

        valid_firms = []
        for j, i_ids in j_i_ids.items():
            if len(set(i_ids)) >= threshold:
                valid_firms.append(j)

        return np.array(valid_firms)

    def min_workers_frame(self, threshold=15, drop_returns_to_stays=False, is_sorted=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many workers.

        Arguments:
            threshold (int): minimum number of workers required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe of firms with sufficiently many workers
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
        no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

        # Compute min_workers_frame
        frame = self.to_long(drop_no_split_columns=False, is_sorted=is_sorted, copy=copy).min_workers_frame(threshold=threshold, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, copy=False).to_eventstudy(is_sorted=True, copy=False)

        # Update col_long_es_dict for columns that aren't supposed to convert to long
        for col in no_split_cols:
            frame.col_long_es_dict[col] = None

        return frame

    def min_moves_firms(self, threshold=2, is_sorted=False, copy=True):
        '''
        List firms with at least `threshold` many moves. Note that a single mover can have multiple moves at the same firm.

        Arguments:
            threshold (int): minimum number of moves required to keep a firm
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (NumPy Array): firms with sufficiently many moves
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        return self.to_long(is_sorted=is_sorted, copy=copy).min_moves_firms(threshold=threshold)

    def min_moves_frame(self, threshold=2, drop_returns_to_stays=False, is_sorted=False, reset_index=False, copy=True):
        '''
        Return dataframe where all firms have at least `threshold` many moves. Note that a single worker can have multiple moves at the same firm. This method employs loops, as dropping firms that don't meet the threshold may lower the number of moves at other firms.

        Arguments:
            threshold (int): minimum number of moves required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): not used for event study format
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe of firms with sufficiently many moves
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
        no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

        # Compute min_moves_frame
        frame = self.to_long(drop_no_split_columns=False, is_sorted=is_sorted, copy=copy).min_moves_frame(threshold=threshold, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False).to_eventstudy(is_sorted=True, copy=False)

        # Update col_long_es_dict for columns that aren't supposed to convert to long
        for col in no_split_cols:
            frame.col_long_es_dict[col] = None

        return frame

    def min_movers_frame(self, threshold=15, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Return dataframe where all firms have at least `threshold` many movers. This method employs loops, as dropping firms that don't meet the threshold may lower the number of movers at other firms.

        Arguments:
            threshold (int): minimum number of movers required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Returned dataframe will be sorted. Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
            reset_index (bool): not used for event study format
            copy (bool): if False, avoid copy.

        Returns:
            (BipartiteEventStudyBase): dataframe of firms with sufficiently many movers
        '''
        self.log('generating frame of firms with a minimum number of movers', level='info')

        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        # Keep track of columns that aren't supposed to convert to long, but we allow to convert because this is during data cleaning
        no_split_cols = [col for col, long_es_split in self.col_long_es_dict.items() if long_es_split is None]

        # Compute min_movers_frame
        frame = self.to_long(drop_no_split_columns=False, is_sorted=is_sorted, copy=copy).min_movers_frame(threshold=threshold, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False).to_eventstudy(is_sorted=True, copy=False)

        # Update col_long_es_dict for columns that aren't supposed to convert to long
        for col in no_split_cols:
            frame.col_long_es_dict[col] = None

        return frame

    def construct_artificial_time(self, time_per_worker=False, is_sorted=False, copy=True):
        '''
        Construct artificial time columns to enable conversion to (collapsed) long format. Only adds columns if time columns not already included.

        Arguments:
            time_per_worker (bool): if True, set time independently for each worker (note that this is significantly more computationally costly)
            is_sorted (bool): set to True if dataframe is already sorted by i (this avoids a sort inside a groupby if time_per_worker=True, but this groupby will not sort the returned dataframe)
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe with artificial time columns
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if not frame._col_included('t'):
            # Values for t columns
            if time_per_worker:
                # Reset time for each worker
                t1 = frame.groupby('i', sort=(not is_sorted)).cumcount()
            else:
                # Cumulative time over all workers
                t1 = np.arange(len(frame))
            t2 = t1 + 1
            # t column names
            t_subcols = bpd.util.to_list(frame.col_reference_dict['t'])
            halfway = len(t_subcols) // 2
            for i in range(halfway):
                # Iterate over t columns and fill in values
                frame.loc[:, t_subcols[i]] = t1
                frame.loc[:, t_subcols[i + halfway]] = t2

                # Update time for stayers to be constant
                stayers = ~frame.get_worker_m(is_sorted=is_sorted)
                frame.loc[stayers, t_subcols[i + halfway]] = frame.loc[stayers, t_subcols[i]]

        # Sort columns
        frame = frame.sort_cols(copy=False)

        return frame
