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

        # self.logger.info('BipartiteEventStudyBase object initialized')

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
            self.logger.info("'m' column already included. Returning unaltered frame.")

        return frame

    # def clean_data(self, user_clean={}):
    #     '''
    #     Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

    #     Arguments:
    #         user_clean (dict): dictionary of parameters for cleaning

    #             Dictionary parameters:

    #                 connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'connected_strong' keep observations in the largest strongly connected set of firms; if None, keep all observations

    #                 i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids

    #                 drop_multiples (bool): if True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency when re-collapsing data for biconnected components)

    #                 data_validity (bool, default=True): if True, run data validity checks; much faster if set to False

    #                 copy (bool, default=False): if False, avoid copy

    #     Returns:
    #         frame (BipartiteEventStudyBase): BipartiteEventStudyBase with cleaned data
    #     '''
    #     frame = bpd.BipartiteBase.clean_data(self, user_clean)

    #     if ('data_validity' not in user_clean.keys()) or user_clean['data_validity']:
    #         frame.logger.info('beginning BipartiteEventStudyBase data cleaning')
    #         frame.logger.info('checking quality of data')
    #         frame = frame._data_validity()

    #     frame.logger.info('BipartiteEventStudyBase data cleaning complete')

    #     return frame

    # def _data_validity(self):
    #     '''
    #     Checks that data is formatted correctly and updates relevant attributes.

    #     Returns:
    #         frame (BipartiteEventStudyBase): BipartiteEventStudyBase with corrected columns and attributes
    #     '''
    #     frame = self # .copy()

    #     success_stayers = True
    #     success_movers = True

    #     stayers = frame.loc[frame.loc[:, 'm'].to_numpy() == 0, :]
    #     movers = frame.loc[frame.loc[:, 'm'].to_numpy() > 0, :]

    #     frame.logger.info('--- checking firms ---')
    #     firms_stayers = (stayers.loc[:, 'j1'].to_numpy() != stayers.loc[:, 'j2'].to_numpy()).sum()
    #     firms_movers = (movers.loc[:, 'j1'].to_numpy() == movers.loc[:, 'j2'].to_numpy()).sum()

    #     frame.logger.info('stayers with different firms (should be 0):' + str(firms_stayers))
    #     frame.logger.info('movers with same firm (should be 0):' + str(firms_movers))
    #     if firms_stayers > 0:
    #         success_stayers = False
    #     if firms_movers > 0:
    #         success_movers = False

    #     frame.logger.info('--- checking income ---')
    #     income_stayers = (stayers.loc[:, 'y1'].to_numpy() != stayers.loc[:, 'y2'].to_numpy()).sum()

    #     frame.logger.info('stayers with different income (should be 0):' + str(income_stayers))
    #     if income_stayers > 0:
    #         success_stayers = False

    #     frame.logger.info('Overall success for stayers:' + str(success_stayers))
    #     frame.logger.info('Overall success for movers:' + str(success_movers))

    #     return frame

    def _drop_i_t_duplicates(self, how='max', copy=True):
        '''
        Keep only the highest paying job for i-t (worker-year) duplicates.

        Arguments:
            how (str): if 'max', keep max paying job; otherwise, take `how` over duplicate worker-firm-year observations, then take the highest paying worker-firm observation. `how` can take any option valid for a Pandas transform. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids
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
            # Note: we use unstack rather than convert to long because duplicates mean that we should fully unstack all observations, to see which are duplicates and which are legitimate - converting to long will arbitrarily decide which rows are already correct
            frame = frame.unstack_es(is_clean=False).drop_duplicates()
            if isinstance(frame, bpd.BipartiteLongCollapsed):
                # If self is BipartiteEventStudyCollapsed, then frame is BipartiteLongCollapsed
                frame = frame.uncollapse()
                collapsed = True
            else:
                collapsed = False

            frame = bpd.BipartiteBase._drop_i_t_duplicates(frame, how, copy=False)

            # Return to event study
            if collapsed:
                frame = frame.get_collapsed_long(copy=False)
            frame = frame.get_es()

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

        self.logger.info('mover and stayer event study datasets combined into cross section')

        return data_cs

    def get_long(self, is_clean=True, return_df=False):
        '''
        Return (collapsed) event study data reformatted into (collapsed) long form.

        Arguments:
            is_clean (bool): if True, data is already clean
            return_df (bool): if True, return a Pandas dataframe instead of a BipartiteLong(Collapsed) dataframe

        Returns:
            long_frame (BipartiteLong(Collapsed) or Pandas DataFrame): BipartiteLong(Collapsed) or Pandas dataframe generated from (collapsed) event study data
        '''
        # Dictionary to swap names (necessary for last row of data, where period-2 observations are not located in subsequent period-1 column (as it doesn't exist), so must append the last row with swapped column names)
        rename_dict_1 = {}
        # Dictionary to reformat names into (collapsed) long form
        rename_dict_2 = {}
        # For casting column types
        astype_dict = {}
        # Columns to drop
        drops = []
        for col in self._included_cols():
            subcols = bpd.to_list(self.reference_dict[col])
            n_subcols = len(subcols)
            # If even number of subcols, then is formatted as 'x1', 'x2', etc., so must swap to be 'x2', 'x1', etc.
            if n_subcols % 2 == 0:
                halfway = n_subcols // 2
                for i in range(halfway):
                    rename_dict_1[subcols[i]] = subcols[halfway + i]
                    rename_dict_1[subcols[halfway + i]] = subcols[i]
                    subcol_number = subcols[i].strip(col) # E.g. j1 will give 1
                    rename_dict_2[subcols[i]] = col + subcol_number[1:] # Get rid of first number, e.g. j12 to j2 (note there is no indexing issue even if subcol_number has only one digit)
                    if self.col_dtype_dict[col] == 'int':
                        astype_dict[rename_dict_2[subcols[i]]] = int

                    drops.append(subcols[halfway + i])

            else:
                # Check correct type for other columns
                if self.col_dtype_dict[col] == 'int':
                    astype_dict[col] = int

        # Sort by i, t if t included; otherwise sort by i
        sort_order_1 = ['i']
        sort_order_2 = ['i']
        if self._col_included('t'):
            sort_order_1.append(bpd.to_list(self.reference_dict['t'])[0]) # Pre-reformatting
            sort_order_2.append(bpd.to_list(self.reference_dict['t'])[0][: -1]) # Remove last number, e.g. t11 to t1

        if is_clean:
            # Append the last row if a mover (this is because the last observation is only given as an f2i, never as an f1i)
            # Details: if a worker's last observation is a move OR if a worker switches from a move to a stay between observations, we need to append the second observation from the move
            i_col = self.loc[:, 'i'].to_numpy()
            m_col = (self.loc[:, 'm'].to_numpy() > 0)
            i_next = np.roll(i_col, -1)
            m_next = np.roll(m_col, -1)
            m_change = (i_col == i_next) & (m_col) & (~m_next)
            # Source: https://stackoverflow.com/a/47115520/17333120
            i_last = (i_col != i_next)
            last_obs_df = pd.DataFrame(self.loc[(m_change) | (i_last & m_col), :]) \
                .sort_values(sort_order_1) \
                .drop_duplicates(subset='i', keep='last') \
                .rename(rename_dict_1, axis=1) # Sort by i, t to ensure last observation is actually last
        else:
            # If data isn't clean, just stack all moves and deal with duplicates later
            last_obs_df = pd.DataFrame(self.loc[self.loc[:, 'm'].to_numpy() > 0, :]) \
                .sort_values(sort_order_1) \
                .drop_duplicates(subset='i', keep='last') \
                .rename(rename_dict_1, axis=1) # Sort by i, t to ensure last observation is actually last

        try:
            data_long = pd.concat([pd.DataFrame(self), last_obs_df], ignore_index=True) \
                .drop(drops, axis=1) \
                .rename(rename_dict_2, axis=1) \
                .astype(astype_dict, copy=False)
        except ValueError: # If nan values, use Int8
            for col in astype_dict.keys():
                astype_dict[col] = 'Int64'
            data_long = pd.concat([pd.DataFrame(self), last_obs_df], ignore_index=True) \
                .drop(drops, axis=1) \
                .rename(rename_dict_2, axis=1) \
                .astype(astype_dict, copy=False)
        # data_long = pd.DataFrame(self).groupby('i').apply(lambda a: a.append(a.iloc[-1].rename(rename_dict_1, axis=1)) if a.iloc[0]['m'] == 1 else a) \
        #     .reset_index(drop=True) \
        #     .drop(drops, axis=1) \
        #     .rename(rename_dict_2, axis=1) \
        #     .astype(astype_dict, copy=False)

        # Sort columns and rows
        sorted_cols = sorted(data_long.columns, key=bpd.col_order)
        data_long = data_long.reindex(sorted_cols, axis=1, copy=False)
        data_long.sort_values(sort_order_2, inplace=True)
        data_long.reset_index(drop=True, inplace=True)

        if return_df:
            return data_long

        long_frame = self._constructor_long(data_long)
        long_frame._set_attributes(self, no_dict=True)
        long_frame = long_frame.gen_m(force=True, copy=False)

        return long_frame

    def unstack_es(self, is_clean=True, return_df=False):
        '''
        Unstack (collapsed) event study data by stacking (and renaming) period 2 data below period 1 data for movers, then dropping period 2 columns, returning a (collapsed) long dataframe. Duplicates created from unstacking are dropped.

        Arguments:
            is_clean (bool): if True, data is already clean
            return_df (bool): if True, return a Pandas dataframe instead of a BipartiteLong(Collapsed) dataframe

        Returns:
            long_frame (BipartiteLong(Collapsed) or Pandas DataFrame): BipartiteLong(Collapsed) or Pandas dataframe generated from (collapsed) event study data
        '''
        # Dictionary to swap names (necessary for last row of data, where period-2 observations are not located in subsequent period-1 column (as it doesn't exist), so must append the last row with swapped column names)
        rename_dict_1 = {}
        # Dictionary to reformat names into (collapsed) long form
        rename_dict_2 = {}
        # For casting column types
        astype_dict = {}
        # Columns to drop
        drops = []
        for col in self._included_cols():
            subcols = bpd.to_list(self.reference_dict[col])
            n_subcols = len(subcols)
            # If even number of subcols, then is formatted as 'x1', 'x2', etc., so must swap to be 'x2', 'x1', etc.
            if n_subcols % 2 == 0:
                halfway = n_subcols // 2
                for i in range(halfway):
                    rename_dict_1[subcols[i]] = subcols[halfway + i]
                    rename_dict_1[subcols[halfway + i]] = subcols[i]
                    subcol_number = subcols[i].strip(col) # E.g. j1 will give 1
                    rename_dict_2[subcols[i]] = col + subcol_number[1:] # Get rid of first number, e.g. j12 to j2 (note there is no indexing issue even if subcol_number has only one digit)
                    if self.col_dtype_dict[col] == 'int':
                        astype_dict[rename_dict_2[subcols[i]]] = int

                    drops.append(subcols[halfway + i])

            else:
                # Check correct type for other columns
                if self.col_dtype_dict[col] == 'int':
                    astype_dict[col] = int

        # Sort by i, t if t included; otherwise sort by i
        sort_order = ['i']
        if self._col_included('t'):
            sort_order.append(bpd.to_list(self.reference_dict['t'])[0][: - 1]) # Remove last number, e.g. t11 to t1

        if is_clean:
            # Stack period 2 data if a mover (this is because the last observation is only given as an f2i, never as an f1i)
            # Details: if a worker's last observation is a move OR if a worker switches from a move to a stay between observations, we need to append the second observation from the move
            i_col = self.loc[:, 'i'].to_numpy()
            m_col = (self.loc[:, 'm'].to_numpy() > 0)
            i_next = np.roll(i_col, -1)
            m_next = np.roll(m_col, -1)
            m_change = (i_col == i_next) & (m_col) & (~m_next)
            # Source: https://stackoverflow.com/a/47115520/17333120
            i_last = (i_col != i_next)
            stacked_df = pd.DataFrame(self.loc[(m_change) | (i_last & m_col), :]).rename(rename_dict_1, axis=1)
        else:
            # If data isn't clean, just stack all moves and deal with duplicates later
            stacked_df = pd.DataFrame(self.loc[self.loc[:, 'm'].to_numpy() > 0, :]).rename(rename_dict_1, axis=1)

        try:
            data_long = pd.concat([pd.DataFrame(self), stacked_df], ignore_index=True) \
                .drop(drops, axis=1) \
                .rename(rename_dict_2, axis=1) \
                .astype(astype_dict, copy=False)
        except ValueError: # If nan values, use Int8
            for col in astype_dict.keys():
                astype_dict[col] = 'Int64'
            data_long = pd.concat([pd.DataFrame(self), stacked_df], ignore_index=True) \
                .drop(drops, axis=1) \
                .rename(rename_dict_2, axis=1) \
                .astype(astype_dict, copy=False)

        # Sort columns and rows
        sorted_cols = sorted(data_long.columns, key=bpd.col_order)
        data_long = data_long.reindex(sorted_cols, axis=1, copy=False)
        data_long.sort_values(sort_order, inplace=True)
        data_long.reset_index(drop=True, inplace=True)

        if return_df:
            return data_long

        long_frame = self._constructor_long(data_long)
        long_frame._set_attributes(self, no_dict=True)
        long_frame = long_frame.gen_m(force=True, copy=False)

        return long_frame

    def _leave_one_observation_out(self, cc_list, how_max='length', drop_multiples=False):
        '''
        Extract largest leave-one-observation-out connected component.

        Arguments:
            cc_list (list of lists): each entry is a connected component
            how_max (str): how to determine largest biconnected component. Options are 'length', 'firms', and 'workers', where each option chooses the biconnected component with the highest of the chosen value
            drop_multiples (bool): if True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency when re-collapsing data)

        Returns:
            frame_largest_cc (BipartiteEventStudyBase): dataframe of largest leave-one-observation-out connected component
        '''
        return self.get_long()._leave_one_observation_out(cc_list=cc_list, how_max=how_max, drop_multiples=drop_multiples).get_es()

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
        i_next = np.roll(i_col, -1)
        j2_next = np.roll(base_linkages[:, 1], -1)
        valid_i = (i_col == i_next)
        secondary_linkages = np.stack([base_linkages[valid_i, 1], j2_next[valid_i]], axis=1)
        linkages = np.concatenate([base_linkages, secondary_linkages], axis=0)
        return linkages

    def keep_ids(self, id_col, keep_ids, drop_multiples=False, reset_index=False, copy=True):
        '''
        Only keep ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            keep_ids (list): ids to keep
            drop_multiples (bool): used only if id_col == 'j' and using BipartiteEventStudyCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            reset_index (bool): used for long format, does nothing for event study
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteEventStudyBase): dataframe with ids in the given set
        '''
        keep_ids = set(keep_ids)
        if len(keep_ids) == self.n_unique_ids(id_col):
            # If keeping everything
            if copy:
                return self.copy()
            return self

        return self.get_long().keep_ids(id_col=id_col, keep_ids=keep_ids, drop_multiples=drop_multiples, reset_index=False, copy=copy).get_es()

    def drop_ids(self, id_col, drop_ids, drop_multiples=False, reset_index=False, copy=True):
        '''
        Drop ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            drop_ids (list): ids to drop
            drop_multiples (bool): used only if id_col == 'j' and using BipartiteEventStudyCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            reset_index (bool): used for long format, does nothing for event study
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteEventStudyBase): dataframe with ids outside the given set
        '''
        drop_ids = set(drop_ids)
        if len(drop_ids) == 0:
            # If nothing input
            if copy:
                return self.copy()
            return self

        return self.get_long().drop_ids(id_col=id_col, drop_ids=drop_ids, drop_multiples=drop_multiples, reset_index=False, copy=copy).get_es()

    def keep_rows(self, rows, drop_multiples=False, reset_index=False, copy=True):
        '''
        Only keep particular rows.

        Arguments:
            rows (list): rows to keep
            drop_multiples (bool): used only if using BipartiteEventStudyCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
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

        return self.get_long().keep_rows(rows=rows, drop_multiples=drop_multiples, reset_index=False, copy=copy).get_es()

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

        # We consider j1 for all rows, but j2 only for the last row for movers
        j_obs = self.loc[:, 'j1'].value_counts().to_dict()
        # Details: if a worker's last observation is a move OR if a worker switches from a move to a stay between observations, we need to append the second observation from the move
        i_col = self.loc[:, 'i'].to_numpy()
        m_col = (self.loc[:, 'm'].to_numpy() > 0)
        i_next = np.roll(i_col, -1)
        m_next = np.roll(m_col, -1)
        m_change = (i_col == i_next) & (m_col) & (~m_next)
        # Source: https://stackoverflow.com/a/47115520/17333120
        i_last = (i_col != i_next)
        # Get last observation per worker
        j2_obs = self.loc[(m_change) | (i_last & m_col), 'j2'].value_counts().to_dict()
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

    def min_obs_frame(self, threshold=2, drop_multiples=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many observations.

        Arguments:
            threshold (int): minimum number of observations required to keep a firm
            drop_multiples (bool): used only for BipartiteEventStudyCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe of firms with sufficiently many observations
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        return self.get_long().min_obs_frame(threshold=threshold, drop_multiples=drop_multiples, copy=copy).get_es()

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

        # We consider j1 for all rows, but j2 only for the last row for movers
        j_i_ids = self.groupby('j1')['i'].unique().apply(list).to_dict()
        j2_i_ids = self.loc[self.loc[:, 'm'].to_numpy() > 0, :].groupby('i')['j2'].last().reset_index().groupby('j2')['i'].unique().apply(list).to_dict()
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

    def min_workers_frame(self, threshold=15, drop_multiples=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many workers.

        Arguments:
            threshold (int): minimum number of workers required to keep a firm
            drop_multiples (bool): used only for BipartiteEventStudyCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteEventStudyBase): dataframe of firms with sufficiently many workers
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        return self.get_long().min_workers_frame(threshold=threshold, drop_multiples=drop_multiples, copy=copy).get_es()

    def min_moves_firms(self, threshold=2):
        '''
        List firms with at least `threshold` many moves. Note that a single mover can have multiple moves at the same firm.

        Arguments:
            threshold (int): minimum number of moves required to keep a firm

        Returns:
            valid_firms (NumPy Array): firms with sufficiently many moves
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        return self.get_long().min_moves_firms(threshold=threshold)

    def min_moves_frame(self, threshold=2, drop_multiples=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many moves. Note that a single mover can have multiple moves at the same firm.

        Arguments:
            threshold (int): minimum number of moves required to keep a firm
            drop_multiples (bool): used only for collapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteBase): dataframe of firms with sufficiently many moves
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        return self.get_long().min_moves_frame(threshold=threshold, drop_multiples=drop_multiples, copy=copy).get_es()

    # def min_moves_firms(self, threshold=2):
    #     '''
    #     List firms with at least `threshold` many moves. Note that a single mover can have multiple moves at the same firm. Also note that if a worker moves to a firm then leaves it, that counts as two moves - even if the worker was at the firm for only one period.

    #     Arguments:
    #         threshold (int): minimum number of moves required to keep a firm

    #     Returns:
    #         valid_firms (NumPy Array): firms with sufficiently many moves
    #     '''
    #     if threshold == 0:
    #         # If no threshold
    #         return self.unique_ids('j')

    #     # We consider j1 for all rows, but j2 only for the last row for movers
    #     j_moves = self.groupby('j1')['m'].sum().to_dict()
    #     j2_moves = self.loc[self.loc[:, 'm'].to_numpy() > 0, :].groupby('i')[['j2', 'm']].last().reset_index().groupby('j2')['m'].sum().to_dict() # self.groupby('j2')['m'].sum().to_dict()
    #     for j, n_moves in j2_moves.items():
    #         try:
    #             # If firm j in j1, add the worker ids in j2
    #             j_moves[j] += n_moves
    #         except KeyError:
    #             # If firm j not in j1, set worker ids to be j2
    #             j_moves[j] = n_moves

    #     valid_firms = []
    #     for j, n_moves in j_moves.items():
    #         if n_moves >= threshold:
    #             valid_firms.append(j)

    #     return np.array(valid_firms)

    # def min_moves_frame(self, threshold=2, drop_multiples=False, copy=True):
    #     '''
    #     Return dataframe of firms with at least `threshold` many moves. Note that a single mover can have multiple moves at the same firm.

    #     Arguments:
    #         threshold (int): minimum number of moves required to keep a firm
    #         drop_multiples (bool): used only for BipartiteEventStudyCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
    #         copy (bool): if False, avoid copy

    #     Returns:
    #         (BipartiteEventStudyBase): dataframe of firms with sufficiently many moves
    #     '''
    #     if threshold == 0:
    #         # If no threshold
    #         if copy:
    #             return self.copy()
    #         return self

    #     return self.get_long().min_moves_frame(threshold, drop_multiples, copy).get_es()
