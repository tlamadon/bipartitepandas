'''
Base class for bipartite networks in event study or collapsed event study form
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import igraph as ig

class BipartiteEventStudyBase(bpd.BipartiteBase):
    '''
    Base class for BipartiteEventStudy and BipartiteEventStudyCollapsed, where BipartiteEventStudy and BipartiteEventStudyCollapsed give a bipartite network of firms and workers in event study and collapsed event study form, respectively. Contains generalized methods. Inherits from BipartiteBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        columns_req (list): required columns (only put general column names for joint columns, e.g. put 'fid' instead of 'f1i', 'f2i'; then put the joint columns in reference_dict)
        columns_opt (list): optional columns (only put general column names for joint columns, e.g. put 'g' instead of 'g1', 'g2'; then put the joint columns in reference_dict)
        columns_contig (dictionary): columns requiring contiguous ids linked to boolean of whether those ids are contiguous, or None if column(s) not included, e.g. {'i': False, 'j': False, 'g': None} (only put general column names for joint columns)
        reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'i': 'i', 'j': ['j1', 'j2']}
        col_dtype_dict (dict): link column to datatype
        col_dict (dict or None): make data columns readable. Keep None if column names already correct
        include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, columns_req=[], columns_opt=[], columns_contig={}, reference_dict={}, col_dtype_dict={}, col_dict=None, include_id_reference_dict=False, **kwargs):
        if 't' not in columns_opt:
            columns_opt = ['t'] + columns_opt
        reference_dict = bpd.update_dict({'j': ['j1', 'j2'], 'y': ['y1', 'y2'], 'g': ['g1', 'g2']}, reference_dict)
        # Initialize DataFrame
        super().__init__(*args, columns_req=columns_req, columns_opt=columns_opt, columns_contig=columns_contig, reference_dict=reference_dict, col_dtype_dict=col_dtype_dict, col_dict=col_dict, include_id_reference_dict=include_id_reference_dict, **kwargs)

        # self.log('BipartiteEventStudyBase object initialized', level='info')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteEventStudyBase

    def gen_m(self, force=False, copy=True):
        '''
        Generate m column for data (m == 0 if stayer, m == 1 or 2 if mover).

        Arguments:
            force (bool): if True, reset 'm' column even if it exists
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteBase): BipartiteBase with m column
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if not frame._col_included('m') or force:
            frame.loc[:, 'm'] = (frame.loc[:, 'j1'].to_numpy() != frame.loc[:, 'j2'].to_numpy()).astype(int, copy=False)
            # frame.loc[:, 'm'] = frame.groupby('i')['m'].transform('max')

            frame.col_dict['m'] = 'm'

            # Sort columns
            frame = frame.sort_cols(copy=False)

        else:
            self.log("'m' column already included. Returning unaltered frame.", level='info')

        return frame

    def _get_unstack_rows(self):
        '''
        Get mask of rows where the second observation isn't included in the first observation for any rows (e.g., the last observation for a mover is given only as an j2, never as j1). More details: if a worker's last observation is a move OR if a worker switches from a move to a stay between observations, we need to append the second observation from the move. We can use the fact that the time changes between observations to indicate that we need to unstack, because event studies are supposed to go A -> B, B -> C, so if it goes A -> B, C -> D that means B should be unstacked.

        Returns:
            (NumPy Array): mask of rows to unstack
        '''
        # Get i for this and next period
        i_col = self.loc[:, 'i'].to_numpy()
        i_next = bpd.fast_shift(i_col, -1, fill_value=-2)
        # Get m
        m_col = (self.loc[:, 'm'].to_numpy() > 0)
        # Get t for this and next period
        t_cols = bpd.to_list(self.reference_dict['t'])
        halfway = len(t_cols) // 2
        t1 = t_cols[0]
        t2 = t_cols[halfway]
        t1_next = np.roll(self.loc[:, t1].to_numpy(), -1)
        t2_col = self.loc[:, t2].to_numpy()
        # Check if t changed
        t_change = (i_col == i_next) & (t1_next != t2_col)
        # Check if i changed
        # Source: https://stackoverflow.com/a/47115520/17333120
        i_last = (i_col != i_next)

        return m_col & (t_change | i_last)

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

            ret_str += 'm==0 with different firms (should be 0): {}\n'.format(firms_stayers)
            ret_str += 'm>0 with same firm (should be 0): {}\n'.format(firms_movers)

            ##### Income #####
            income_stayers = (stayers.loc[:, 'y1'].to_numpy() != stayers.loc[:, 'y2'].to_numpy()).sum()

            ret_str += 'm==0 with different income (should be 0): {}'.format(income_stayers)

            print(ret_str)

    def _drop_i_t_duplicates(self, how='max', is_sorted=False, copy=True):
        '''
        Keep only the highest paying job for i-t (worker-year) duplicates.

        Arguments:
            how (str): if 'max', keep max paying job; otherwise, take `how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `how` can take any option valid for a Pandas transform. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteEventStudyBase): dataframe that keeps only the highest paying job for i-t (worker-year) duplicates. If no t column(s), returns frame with no changes
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if frame._col_included('t'):
            # Convert to long
            # Note: we use is_clean=False because duplicates mean that we should fully unstack all observations, to see which are duplicates and which are legitimate - setting is_clean=True would arbitrarily decide which rows are already correct
            frame = frame.get_long(is_clean=False, is_sorted=is_sorted, copy=False)
            frame.drop_duplicates(inplace=True)
            frame = frame._drop_i_t_duplicates(how, is_sorted=True, copy=False).get_es(is_sorted=True, copy=False)

        # Data now has unique i-t observations
        frame.i_t_unique = True

        return frame

    def get_cs(self):
        '''
        Return (collapsed) event study data reformatted into cross section data.

        Returns:
            data_cs (Pandas DataFrame): cross section data
        '''
        sdata = pd.DataFrame(self.loc[self.loc[:, 'm'].to_numpy() == 0, :])
        jdata = pd.DataFrame(self.loc[self.loc[:, 'm'].to_numpy() > 0, :])

        # Columns used for constructing cross section
        cs_cols = self._included_cols(flat=True)

        # Dictionary to swap names for cs=0 (these rows contain period-2 data for movers, so must swap columns for all relevant information to be contained in the same column (e.g. must move y2 into y1, otherwise bottom rows are just duplicates))
        rename_dict = {}
        for col in self._included_cols():
            subcols = bpd.to_list(self.reference_dict[col])
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
        sorted_cols = sorted(data_cs.columns, key=bpd.col_order)
        data_cs = data_cs.reindex(sorted_cols, axis=1, copy=False)

        self.log('mover and stayer event study datasets combined into cross section', level='info')

        return data_cs

    def get_long(self, is_clean=True, is_sorted=False, copy=True):
        '''
        Return (collapsed) event study data reformatted into (collapsed) long form.

        Arguments:
            is_clean (bool): if True, data is already clean (this ensures that observations that are in two consecutive event studies appear only once, e.g. the event study A -> B, B -> C turns into A -> B -> C; otherwise, it will become A -> B -> B -> C). Set to False if duplicates will be handled manually.
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            long_frame (BipartiteLong(Collapsed) or Pandas DataFrame): BipartiteLong(Collapsed) or Pandas dataframe generated from (collapsed) event study data
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        frame = frame.sort_rows(is_sorted=is_sorted, copy=False)

        # Dictionary to swap names (necessary for last row of data, where period-2 observations are not located in subsequent period-1 column (as it doesn't exist), so must append the last row with swapped column names)
        rename_dict_1 = {}
        # Dictionary to reformat names into (collapsed) long form
        rename_dict_2 = {}
        # For casting column types
        astype_dict = {}
        # Columns to drop
        drops = []
        for col in frame._included_cols():
            subcols = bpd.to_list(frame.reference_dict[col])
            n_subcols = len(subcols)
            # If even number of subcols, then is formatted as 'x1', 'x2', etc., so must swap to be 'x2', 'x1', etc.
            if n_subcols % 2 == 0:
                halfway = n_subcols // 2
                for i in range(halfway):
                    rename_dict_1[subcols[i]] = subcols[halfway + i]
                    rename_dict_1[subcols[halfway + i]] = subcols[i]
                    # Get column number, e.g. j1 will give 1
                    subcol_number = subcols[i].strip(col)
                    # Get rid of first number, e.g. j12 to j2 (note there is no indexing issue even if subcol_number has only one digit)
                    rename_dict_2[subcols[i]] = col + subcol_number[1:]
                    if frame.col_dtype_dict[col] == 'int':
                        astype_dict[rename_dict_2[subcols[i]]] = int

                    drops.append(subcols[halfway + i])

            else:
                # Check correct type for other columns
                if frame.col_dtype_dict[col] == 'int':
                    astype_dict[col] = int

        if is_clean:
            # Find rows to unstack
            unstack_df = pd.DataFrame(frame.loc[frame._get_unstack_rows(), :])
        else:
            # If data isn't clean, just unstack all moves and deal with duplicates later
            unstack_df = pd.DataFrame(frame.loc[frame.loc[:, 'm'].to_numpy() > 0, :])
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

        ## Sort columns and rows
        # Sort columns
        sorted_cols = sorted(data_long.columns, key=bpd.col_order)
        data_long = data_long.reindex(sorted_cols, axis=1, copy=False)
        # Sort rows by i (and t, if included)
        sort_order = ['i']
        if frame._col_included('t'):
            # Remove last number, e.g. t11 to t1
            sort_order.append(bpd.to_list(frame.reference_dict['t'])[0][: -1])
        data_long.sort_values(sort_order, inplace=True)
        data_long.reset_index(drop=True, inplace=True)

        long_frame = frame._constructor_long(data_long)
        long_frame._set_attributes(frame, no_dict=True)
        long_frame = long_frame.gen_m(force=True, copy=False)

        return long_frame

    def _drop_returns(self, how=False, is_sorted=False, reset_index=False, copy=True):
        '''
        Drop observations where workers leave a firm then return to it.

        Arguments:
            how (str): if 'returns', drop observations where workers leave a firm then return to it; if 'returners', drop workers who ever leave then return to a firm
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): used for long format, does nothing for event study
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteEventStudyBase): dataframe that drops observations where workers leave a firm then return to it
        '''
        return self.get_long(is_sorted=is_sorted, copy=copy)._drop_returns(how=how, is_sorted=True, reset_index=False, copy=False).get_es(is_sorted=True, copy=False)

    def _prep_cluster(self, stayers_movers=None, t=None, weighted=True, is_sorted=False, copy=False):
        '''
        Prepare data for clustering.

        Arguments:
            stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers
            t (int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)
            weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy
        Returns:
            data (Pandas DataFrame): data prepared for clustering
            weights (NumPy Array or None): if weighted=True, gives NumPy array of firm weights for clustering; otherwise, is None
            jids (NumPy Array): firm ids of firms in subset of data used to cluster
        '''
        return self.get_long(is_sorted=is_sorted, copy=copy)._prep_cluster(stayers_movers=stayers_movers, t=t, weighted=weighted, copy=False)

    def _leave_one_observation_out(self, cc_list, component_size_variable='length', drop_returns_to_stays=False):
        '''
        Extract largest leave-one-observation-out connected component.

        Arguments:
            cc_list (list of lists): each entry is a connected component
            component_size_variable (str): how to determine largest leave-one-observation-out connected component. Options are 'len'/'length' (length of frame), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), and 'movers' (number of unique movers)
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)

        Returns:
            frame_largest_cc (BipartiteEventStudyBase): dataframe of largest leave-one-observation-out connected component
        '''
        return self.get_long(is_sorted=True, copy=False)._leave_one_observation_out(cc_list=cc_list, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays).get_es(is_sorted=True, copy=False)

    def _leave_one_firm_out(self, bcc_list, component_size_variable='length', drop_returns_to_stays=False):
        '''
        Extract largest leave-one-firm-out connected component.

        Arguments:
            bcc_list (list of lists): each entry is a biconnected component
            component_size_variable (str): how to determine largest leave-one-firm-out connected component. Options are 'len'/'length' (length of frame), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), and 'movers' (number of unique movers)
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)

        Returns:
            frame_largest_bcc (BipartiteEventStudyBase): dataframe of largest leave-one-out connected component
        '''
        return self.get_long(is_sorted=True, copy=False)._leave_one_firm_out(bcc_list=bcc_list, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays).get_es(is_sorted=True, copy=False)

    def _construct_connected_linkages(self):
        '''
        Construct numpy array linking firms by movers, for use with connected components.

        Returns:
            (NumPy Array): firm linkages
        '''
        return self.loc[(self.loc[:, 'm'].to_numpy() > 0), ['j1', 'j2']].to_numpy()

    def _construct_biconnected_linkages(self):
        '''
        Construct numpy array linking firms by movers, for use with biconnected components.

        Returns:
            (NumPy Array): firm linkages
        '''
        move_rows = (self.loc[:, 'm'].to_numpy() > 0)
        base_linkages = self.loc[move_rows, ['j1', 'j2']].to_numpy()
        i_col = self.loc[move_rows, 'i'].to_numpy()
        i_next = bpd.fast_shift(i_col, -1, fill_value=-2)
        j2_next = np.roll(base_linkages[:, 1], -1)
        valid_i = (i_col == i_next)
        secondary_linkages = np.stack([base_linkages[valid_i, 1], j2_next[valid_i]], axis=1)
        linkages = np.concatenate([base_linkages, secondary_linkages], axis=0)
        return linkages

    def keep_ids(self, id_col, keep_ids_list, drop_returns_to_stays=False, is_sorted=False, reset_index=False, copy=True):
        '''
        Only keep ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            keep_ids_list (list): ids to keep
            drop_returns_to_stays (bool): used only if id_col is 'j' or 'g' and using BipartiteEventStudyCollapsed format. If True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer).
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): used for long format, does nothing for event study
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteEventStudyBase): dataframe with ids in the given set
        '''
        keep_ids_list = set(keep_ids_list)
        if len(keep_ids_list) == self.n_unique_ids(id_col):
            # If keeping everything
            if copy:
                return self.copy()
            return self

        return self.get_long(is_sorted=is_sorted, copy=copy).keep_ids(id_col=id_col, keep_ids_list=keep_ids_list, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False).get_es(is_sorted=True, copy=False)

    def drop_ids(self, id_col, drop_ids_list, drop_returns_to_stays=False, is_sorted=False, reset_index=False, copy=True):
        '''
        Drop ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            drop_ids_list (list): ids to drop
            drop_returns_to_stays (bool): used only if id_col is 'j' or 'g' and using BipartiteEventStudyCollapsed format. If True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer).
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): used for long format, does nothing for event study
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteEventStudyBase): dataframe with ids outside the given set
        '''
        drop_ids_list = set(drop_ids_list)
        if len(drop_ids_list) == 0:
            # If nothing input
            if copy:
                return self.copy()
            return self

        return self.get_long(is_sorted=is_sorted, copy=copy).drop_ids(id_col=id_col, drop_ids_list=drop_ids_list, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False).get_es(is_sorted=True, copy=False)

    def keep_rows(self, rows, drop_returns_to_stays=False, is_sorted=False, reset_index=False, copy=True):
        '''
        Only keep particular rows.

        Arguments:
            rows (list): rows to keep
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): used for long format, does nothing for event study
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteEventStudyBase): dataframe with given rows
        '''
        rows = set(rows)
        if len(rows) == len(self):
            # If keeping everything
            if copy:
                return self.copy()
            return self

        return self.get_long(is_sorted=is_sorted, copy=copy).keep_rows(rows=rows, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False).get_es(is_sorted=True, copy=False)

    def min_obs_firms(self, threshold=2):
        '''
        List firms with at least `threshold` many observations.

        Arguments:
            threshold (int): minimum number of observations required to keep a firm

        Returns:
            valid_firms (NumPy Array): firms with sufficiently many observations
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        # We consider j1 for all rows, but j2 only for unstack rows
        j_obs = self.loc[:, 'j1'].value_counts(sort=False).to_dict()
        j2_obs = self.loc[self._get_unstack_rows(), 'j2'].value_counts(sort=False).to_dict()
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
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe of firms with sufficiently many observations
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        return self.get_long(is_sorted=is_sorted, copy=copy).min_obs_frame(threshold=threshold, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, copy=False).get_es(is_sorted=True, copy=False)

    def min_workers_firms(self, threshold=2):
        '''
        List firms with at least `threshold` many workers.

        Arguments:
            threshold (int): minimum number of workers required to keep a firm

        Returns:
            valid_firms (NumPy Array): firms with sufficiently many workers
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        # We consider j1 for all rows, but j2 only for unstack rows
        j_i_ids = self.groupby('j1')['i'].unique().apply(list).to_dict()
        j2_i_ids = self.loc[self._get_unstack_rows(), :].groupby('j2')['i'].unique().apply(list).to_dict()
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
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe of firms with sufficiently many workers
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        return self.get_long(is_sorted=is_sorted, copy=copy).min_workers_frame(threshold=threshold, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, copy=False).get_es(is_sorted=True, copy=False)

    def min_moves_firms(self, threshold=2, is_sorted=False, copy=True):
        '''
        List firms with at least `threshold` many moves. Note that a single mover can have multiple moves at the same firm.

        Arguments:
            threshold (int): minimum number of moves required to keep a firm
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            valid_firms (NumPy Array): firms with sufficiently many moves
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        return self.get_long(is_sorted=is_sorted, copy=copy).min_moves_firms(threshold=threshold)

    def min_moves_frame(self, threshold=2, drop_returns_to_stays=False, is_sorted=False, reset_index=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many moves. Note that a single mover can have multiple moves at the same firm.

        Arguments:
            threshold (int): minimum number of moves required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): used for long format, does nothing for event study
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteBase): dataframe of firms with sufficiently many moves
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        return self.get_long(is_sorted=is_sorted, copy=copy).min_moves_frame(threshold=threshold, drop_returns_to_stays=drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False).get_es(is_sorted=True, copy=False)
