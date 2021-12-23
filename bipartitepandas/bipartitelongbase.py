'''
Base class for bipartite networks in long or collapsed long form
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd

class BipartiteLongBase(bpd.BipartiteBase):
    '''
    Base class for BipartiteLong and BipartiteLongCollapsed, where BipartiteLong and BipartiteLongCollapsed give a bipartite network of firms and workers in long and collapsed long form, respectively. Contains generalized methods. Inherits from BipartiteBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        columns_req (list): required columns (only put general column names for joint columns, e.g. put 'fid' instead of 'f1i', 'f2i'; then put the joint columns in reference_dict)
        columns_opt (list): optional columns (only put general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'; then put the joint columns in reference_dict)
        reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}
        col_dtype_dict (dict): link column to datatype
        col_dict (dict or None): make data columns readable. Keep None if column names already correct
        include_id_reference_dict (bool): if True, create dictionary of Pandas dataframes linking original id values to contiguous id values
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, columns_req=[], columns_opt=[], reference_dict={}, col_dtype_dict={}, col_dict=None, include_id_reference_dict=False, **kwargs):
        if 't' not in columns_req:
            columns_req = ['t'] + columns_req
        reference_dict = bpd.update_dict({'j': 'j', 'y': 'y', 'g': 'g'}, reference_dict)
        # Initialize DataFrame
        super().__init__(*args, columns_req=columns_req, columns_opt=columns_opt, reference_dict=reference_dict, col_dtype_dict=col_dtype_dict, col_dict=col_dict, include_id_reference_dict=include_id_reference_dict, **kwargs)

        # self.logger.info('BipartiteLongBase object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteLongBase

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
            i_col = frame.loc[:, 'i'].to_numpy()
            j_col = frame.loc[:, 'j'].to_numpy()
            i_prev = np.roll(i_col, 1)
            i_next = np.roll(i_col, -1)
            j_prev = np.roll(j_col, 1)
            j_next = np.roll(j_col, -1)

            ##### Disable Pandas warning #####
            pd.options.mode.chained_assignment = None
            frame.loc[:, 'm'] = ((i_col == i_prev) & (j_col != j_prev)).astype(int, copy=False) + ((i_col == i_next) & (j_col != j_next)).astype(int, copy=False)
            ##### Re-enable Pandas warning #####
            pd.options.mode.chained_assignment = 'warn'

            frame.col_dict['m'] = 'm'

            # Sort columns
            frame = frame.sort_cols(copy=False)

        else:
            frame.logger.info("'m' column already included. Returning unaltered frame.")

        return frame

    def get_es(self):
        '''
        Return (collapsed) long form data reformatted into (collapsed) event study data.

        Returns:
            es_frame (BipartiteEventStudy(Collapsed)): BipartiteEventStudy(Collapsed) object generated from (collapsed) long data
        '''
        # Split workers by movers and stayers
        stayers = pd.DataFrame(self.loc[self.loc[:, 'm'].to_numpy() == 0, :])
        movers = pd.DataFrame(self.loc[self.loc[:, 'm'].to_numpy() > 0, :])
        self.logger.info('workers split by movers and stayers')

        # Add lagged values
        all_cols = self._included_cols()
        movers = movers.sort_values(['i', bpd.to_list(self.reference_dict['t'])[0]]) # Sort by i, t
        keep_cols = ['i'] # Columns to keep
        for col in all_cols:
            for subcol in bpd.to_list(self.reference_dict[col]):
                subcol_number = subcol.strip(col) # E.g. j1 will give 1
                if subcol != 'm': # Don't want lagged m
                    # Movers
                    plus_1 = col + '1' + subcol_number # Useful for t1 and t2: t1 should go to t11 and t21; t2 should go to t12 and t22
                    plus_2 = col + '2' + subcol_number
                    movers.loc[:, plus_1] = movers.loc[:, subcol].shift(periods=1) # Lagged value
                    movers.rename({subcol: plus_2}, axis=1, inplace=True)
                    # Stayers (no lags)
                    stayers.loc[:, plus_1] = stayers.loc[:, subcol]
                    stayers.rename({subcol: plus_2}, axis=1, inplace=True)
                    if subcol != 'i': # Columns to keep
                        keep_cols += [plus_1, plus_2]
                else:
                    keep_cols.append('m')

        # Ensure lagged values are for the same worker
        # NOTE: cannot use .to_numpy()
        movers = movers.loc[movers.loc[:, 'i1'] == movers.loc[:, 'i2'], :]

        # Correct datatypes (shifting adds nans which converts all columns into float, correct columns that should be int)
        for col in all_cols:
            if (self.col_dtype_dict[col] == 'int') and (col != 'm'):
                for subcol in bpd.to_list(self.reference_dict[col]):
                    subcol_number = subcol.strip(col) # E.g. j1 will give 1
                    movers.loc[:, col + '1' + subcol_number] = movers.loc[:, col + '1' + subcol_number].astype(int, copy=False)

        # Correct i
        movers.drop('i2', axis=1, inplace=True)
        movers.rename({'i1': 'i'}, axis=1, inplace=True)
        stayers.drop('i2', axis=1, inplace=True)
        stayers.rename({'i1': 'i'}, axis=1, inplace=True)

        # Keep only relevant columns
        stayers = stayers.reindex(keep_cols, axis=1, copy=False)
        movers = movers.reindex(keep_cols, axis=1, copy=False)
        self.logger.info('columns updated')

        # Merge stayers and movers
        data_es = pd.concat([stayers, movers], ignore_index=True) # .reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(data_es.columns, key=bpd.col_order)
        data_es = data_es.reindex(sorted_cols, axis=1, copy=False)

        self.logger.info('data reformatted as event study')

        es_frame = self._constructor_es(data_es)
        es_frame._set_attributes(self, no_dict=True)

        # Sort rows
        es_frame.sort_values(['i', bpd.to_list(es_frame.reference_dict['t'])[0]], inplace=True)

        # Recompute 'm'
        es_frame = es_frame.gen_m(force=True, copy=False)

        return es_frame

    def _leave_one_observation_out(self, cc_list, how_max='length', drop_multiples=False):
        '''
        Extract largest leave-one-observation-out connected component.

        Arguments:
            cc_list (list of lists): each entry is a connected component
            how_max (str): how to determine largest biconnected component. Options are 'length', 'firms', and 'workers', where each option chooses the biconnected component with the highest of the chosen value
            drop_multiples (bool): if True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency when re-collapsing data)

        Returns:
            frame_largest_cc (BipartiteLongBase): dataframe of largest leave-one-observation-out connected component
        '''
        # This will become the largest leave-one-observation-out component
        frame_largest_cc = None

        for cc in cc_list:
            # Keep observations in connected components
            frame_cc = self.keep_ids('j', cc, drop_multiples, copy=True)

            if frame_largest_cc is not None:
                # If frame_cc is already smaller than frame_largest_cc
                if how_max == 'length':
                    skip = (len(frame_largest_cc) >= len(frame_cc))
                elif how_max == 'firms':
                    skip = (frame_largest_cc.n_firms() >= frame_cc.n_firms())
                elif how_max == 'workers':
                    skip = (frame_largest_cc.n_workers() >= frame_cc.n_workers())
                else:
                    raise NotImplementedError("Invalid how_max: {}. Valid options are 'length', 'firms', and 'workers'.".format(how_max))
                if skip:
                    break

            # Remove firms with only 1 mover observation (can have 1 mover with multiple observations)
            # This fixes a discrepency between igraph's biconnected components and the definition of leave-one-out connected set, where biconnected components is True if a firm has only 1 mover, since then it disappears from the graph - but leave-one-out requires the set of firms to remain unchanged
            frame_cc = frame_cc.min_moves_frame(2, drop_multiples, copy=False)
            if len(frame_cc) != len(frame_cc.min_moves_frame(2, drop_multiples, copy=True)):
                print('min_moves')
                raise NotImplementedError

            if frame_largest_cc is not None:
                # If frame_cc is already smaller than frame_largest_cc
                if how_max == 'length':
                    skip = (len(frame_largest_cc) >= len(frame_cc))
                elif how_max == 'firms':
                    skip = (frame_largest_cc.n_firms() >= frame_cc.n_firms())
                elif how_max == 'workers':
                    skip = (frame_largest_cc.n_workers() >= frame_cc.n_workers())
                else:
                    raise NotImplementedError("Invalid how_max: {}. Valid options are 'length', 'firms', and 'workers'.".format(how_max))
                if skip:
                    break

            # Construct graph
            G2 = frame_cc._construct_graph('biconnected_observations')

            # Extract articulation firms
            articulation_firms = G2.articulation_points()

            if len(articulation_firms) > 0:
                # If there are articulation firms
                # Extract articulation rows
                articulation_rows = frame_cc._get_articulation_rows(G2, frame_cc.loc[(frame_cc.loc[:, 'j'].isin(articulation_firms)) & (frame_cc.loc[:, 'm'] > 0), :].index.to_numpy())

                if len(articulation_rows) > 0:
                    # If new frame is not leave-one-out connected, drop articulation rows then recompute leave-one-out components
                    frame_cc = frame_cc.drop_rows(articulation_rows, drop_multiples, copy=False)

                    if frame_largest_cc is not None:
                        # If frame_cc is already smaller than frame_largest_cc
                        if how_max == 'length':
                            skip = (len(frame_largest_cc) >= len(frame_cc))
                        elif how_max == 'firms':
                            skip = (frame_largest_cc.n_firms() >= frame_cc.n_firms())
                        elif how_max == 'workers':
                            skip = (frame_largest_cc.n_workers() >= frame_cc.n_workers())
                        else:
                            raise NotImplementedError("Invalid how_max: {}. Valid options are 'length', 'firms', and 'workers'.".format(how_max))
                        if skip:
                            break

                    # Recompute connected components
                    G2 = frame_cc._construct_graph('biconnected_observations')
                    cc_list_2 = G2.components()
                    # Recursion step
                    frame_cc = frame_cc._leave_one_observation_out(cc_list_2, how_max, drop_multiples)
                    if len(frame_cc) != len(frame_cc.min_moves_frame(2, drop_multiples, copy=True)):
                        print('recursion')
                        raise NotImplementedError

            if frame_largest_cc is None:
                # If in the first round
                replace = True
            elif frame_cc is None:
                # If the biconnected components have recursively been eliminated
                replace = False
            else:
                if how_max == 'length':
                    replace = (len(frame_cc) >= len(frame_largest_cc))
                elif how_max == 'firms':
                    replace = (frame_cc.n_firms() >= frame_largest_cc.n_firms())
                elif how_max == 'workers':
                    replace = (frame_cc.n_workers() >= frame_largest_cc.n_workers())
                else:
                    raise NotImplementedError("Invalid how_max: {}. Valid options are 'length', 'firms', and 'workers'.".format(how_max))
            if replace:
                frame_largest_cc = frame_cc

        if len(frame_largest_cc) != len(frame_largest_cc.min_moves_frame(2, drop_multiples, copy=True)):
            print('end')
            raise NotImplementedError

        # Return largest leave-one-observation-out component
        return frame_largest_cc

    def _construct_connected_linkages(self):
        '''
        Construct numpy array linking firms by movers, for use with connected components.

        Returns:
            (NumPy Array): firm linkages
        '''
        move_rows = (self.loc[:, 'm'].to_numpy() > 0)
        i_col = self.loc[move_rows, 'i'].to_numpy()
        j_col = self.loc[move_rows, 'j'].to_numpy()
        j_next = np.roll(j_col, -1)
        i_match = (i_col == np.roll(i_col, -1))
        j_col = j_col[i_match]
        j_next = j_next[i_match]
        linkages = np.stack([j_col, j_next], axis=1)

        return linkages

    def _construct_biconnected_linkages(self):
        '''
        Construct numpy array linking firms by movers, for use with biconnected components.

        Returns:
            (NumPy Array): firm linkages
        '''
        move_rows = (self.loc[:, 'm'].to_numpy() > 0)
        i_col = self.loc[move_rows, 'i'].to_numpy()
        j_col = self.loc[move_rows, 'j'].to_numpy()
        i_next = np.roll(i_col, -1)
        j_next = np.roll(j_col, -1)
        valid_next = (i_col == i_next)
        base_linkages = np.stack([j_col[valid_next], j_next[valid_next]], axis=1)
        i_next_2 = np.roll(i_col, -2)
        j_next_2 = np.roll(j_col, -2)
        valid_next_2 = (i_col == i_next_2)
        secondary_linkages = np.stack([j_col[valid_next_2], j_next_2[valid_next_2]], axis=1)
        linkages = np.concatenate([base_linkages, secondary_linkages], axis=0)
        return linkages

    def _biconnected_linkages_indices(self):
        '''
        Construct numpy array of original indices for biconnected linkages. The first column tells you, for each link in the graph, what index the first observation in the link is coming from; and the second column tells you, for each link in the graph, what index the second observation in the link is coming from.

        Returns:
            (NumPy Array): original indices
        '''
        move_rows = (self.loc[:, 'm'].to_numpy() > 0)
        i_col = self.loc[move_rows, 'i'].to_numpy()
        indices = self.loc[move_rows, :].index.to_numpy()
        i_next = np.roll(i_col, -1)
        indices_next = np.roll(indices, -1)
        valid_next = (i_col == i_next)
        base_indices = np.stack([indices[valid_next], indices_next[valid_next]], axis=1)
        i_next_2 = np.roll(i_col, -2)
        indices_next_2 = np.roll(indices, -2)
        valid_next_2 = (i_col == i_next_2)
        secondary_indices = np.stack([indices[valid_next_2], indices_next_2[valid_next_2]], axis=1)
        original_indices = np.concatenate([base_indices, secondary_indices], axis=0)
        return original_indices

    def _get_articulation_rows(self, G, rows):
        '''
        Compute articulation rows for self, by checking whether self is leave-one-observation-out connected when dropping selected observations one at a time.

        Arguments:
            G (igraph Graph): graph linking firms by movers
            rows (list): list of rows to drop

        Returns:
            (list): articulation rows for self
        '''
        # Get original indices for biconnected linkages
        original_indices = self._biconnected_linkages_indices()
        index_first = original_indices[:, 0]
        index_second = original_indices[:, 1]

        # Save articulation rows (rows that disconnect the graph when they are removed)
        articulation_rows = []

        # Check if each row is an articulation row
        for row in rows:
            G_row = G.copy()
            # Row gives an index in the frame, but we need an index for the graph
            try:
                # If observation is first in pair
                row_indices = list(np.where(index_first == row)[0])
            except IndexError:
                # If observation isn't first in pair
                row_indices = []
            try:
                # If observation is second in pair
                row_indices += list(np.where(index_second == row)[0])
            except:
                # If observation isn't second in pair
                pass

            # Delete row(s)
            # print(G_row.es().get_attribute_values('to'))
            G_row.delete_edges(row_indices)

            # Check whether removing row(s) disconnects graph
            if not G_row.is_connected():
                articulation_rows += [row]

        return articulation_rows

    def keep_ids(self, id_col, keep_ids, drop_multiples=False, reset_index=True, copy=True):
        '''
        Only keep ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            keep_ids (list): ids to keep
            drop_multiples (bool): used only if id_col == 'j' and using BipartiteLongCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteLongBase): dataframe with ids in the given set
        '''
        keep_ids = set(keep_ids)
        if len(keep_ids) == self.n_unique_ids(id_col):
            # If keeping everything
            if copy:
                return self.copy()
            return self

        frame = self.loc[self.loc[:, id_col].isin(keep_ids), :]

        if id_col in ['j', 'g']:
            if isinstance(frame, bpd.BipartiteLongCollapsed):
                # If BipartiteLongCollapsed
                frame = frame.recollapse(drop_multiples=drop_multiples, copy=copy)
                # We don't need to copy again
                copy = False

            # Recompute 'm' since it might change from dropping observations or from re-collapsing
            frame = frame.gen_m(force=True, copy=copy)
            # We don't need to copy again
            copy = False

        if copy:
            # Copy on subset
            frame = frame.copy()

        if reset_index:
            frame.reset_index(drop=True, inplace=True)

        return frame

    def drop_ids(self, id_col, drop_ids, drop_multiples=False, reset_index=True, copy=True):
        '''
        Drop ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            drop_ids (list): ids to drop
            drop_multiples (bool): used only if id_col == 'j' and using BipartiteLongCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteLongBase): dataframe with ids outside the given set
        '''
        drop_ids = set(drop_ids)
        if len(drop_ids) == 0:
            # If nothing input
            if copy:
                return self.copy()
            return self

        frame = self.loc[~(self.loc[:, id_col].isin(drop_ids)), :]

        if id_col in ['j', 'g']:
            if isinstance(frame, bpd.BipartiteLongCollapsed):
                # If BipartiteLongCollapsed
                frame = frame.recollapse(drop_multiples=drop_multiples, copy=copy)
                # We don't need to copy again
                copy = False

            # Recompute 'm' since it might change from dropping observations or from re-collapsing
            frame = frame.gen_m(force=True, copy=copy)
            # We don't need to copy again
            copy = False

        if copy:
            # Copy on subset
            frame = frame.copy()

        if reset_index:
            frame.reset_index(drop=True, inplace=True)

        return frame

    def keep_rows(self, rows, drop_multiples=False, reset_index=True, copy=True):
        '''
        Only keep particular rows.

        Arguments:
            rows (list): rows to keep
            drop_multiples (bool): used only if using BipartiteLongCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteLongBase): dataframe with given rows
        '''
        rows = set(rows)
        if len(rows) == len(self):
            # If keeping everything
            if copy:
                return self.copy()
            return self
        rows = sorted(list(rows))

        frame = self.iloc[rows]

        if isinstance(frame, bpd.BipartiteLongCollapsed):
            # If BipartiteLongCollapsed
            frame = frame.recollapse(drop_multiples=drop_multiples, copy=copy)
            # We don't need to copy again
            copy = False

        # Recompute 'm' since it might change from dropping observations or from re-collapsing
        frame = frame.gen_m(force=True, copy=copy)

        if reset_index:
            frame.reset_index(drop=True, inplace=True)

        return frame

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

        n_obs = self.loc[:, 'j'].value_counts()
        valid_firms = n_obs[n_obs.to_numpy() >= threshold].index.to_numpy()

        return valid_firms

    @bpd.recollapse_loop
    def min_obs_frame(self, threshold=2, drop_multiples=False, copy=True):
        '''
        Interior function for min_obs_frame().

        Arguments:
            threshold (int): minimum number of observations required to keep a firm
            drop_multiples (bool): used only for BipartiteLongCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteLongBase): dataframe of firms with sufficiently many observations
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        frame = self.loc[self.groupby('j')['i'].transform('size').to_numpy() >= threshold, :]

        if isinstance(frame, bpd.BipartiteLongCollapsed):
            # If BipartiteLongCollapsed
            frame = frame.recollapse(drop_multiples=drop_multiples, copy=copy)
            # We don't need to copy again
            copy = False

        # Recompute 'm' since it might change from dropping observations or from re-collapsing
        frame = frame.gen_m(force=True, copy=copy)

        frame.reset_index(drop=True, inplace=True)

        return frame

    def min_workers_firms(self, threshold=15):
        '''
        List firms with at least `threshold` many workers.

        Arguments:
            threshold (int): minimum number of workers required to keep a firm

        Returns:
            valid_firms (NumPy Array): list of firms with sufficiently many workers
        '''
        if threshold == 0:
            # If no threshold
            return self.unique_ids('j')

        n_workers = self.groupby('j')['i'].nunique()
        valid_firms = n_workers[n_workers.to_numpy() >= threshold].index.to_numpy()

        return valid_firms

    @bpd.recollapse_loop
    def min_workers_frame(self, threshold=15, drop_multiples=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many workers.

        Arguments:
            threshold (int): minimum number of workers required to keep a firm
            drop_multiples (bool): used only for BipartiteLongCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteLongBase): dataframe of firms with sufficiently many workers
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        frame = self.loc[self.groupby('j')['i'].transform('nunique').to_numpy() >= threshold, :]

        if isinstance(frame, bpd.BipartiteLongCollapsed):
            # If BipartiteLongCollapsed
            frame = frame.recollapse(drop_multiples=drop_multiples, copy=copy)
            # We don't need to copy again
            copy = False

        # Recompute 'm' since it might change from dropping observations or from re-collapsing
        frame = frame.gen_m(force=True, copy=copy)

        frame.reset_index(drop=True, inplace=True)

        return frame

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

    #     n_moves = self.groupby('j')['m'].sum()
    #     valid_firms = n_moves[n_moves.to_numpy() >= threshold].index.to_numpy()

    #     return valid_firms

    # def min_moves_frame(self, threshold=2, drop_multiples=False, copy=True):
    #     '''
    #     Return dataframe of firms with at least `threshold` many moves. Note that a single mover can have multiple moves at the same firm.

    #     Arguments:
    #         threshold (int): minimum number of moves required to keep a firm
    #         drop_multiples (bool): used only for BipartiteLongCollapsed format. If True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency)
    #         copy (bool): if False, avoid copy

    #     Returns:
    #         (BipartiteLongBase): dataframe of firms with sufficiently many moves
    #     '''
    #     if threshold == 0:
    #         # If no threshold
    #         if copy:
    #             return self.copy()
    #         return self

    #     frame = self.loc[self.groupby('j')['m'].transform('sum').to_numpy() >= threshold, :]

    #    if isinstance(frame, bpd.BipartiteLongCollapsed):
    #        frame = frame.recollapse(drop_multiples=drop_multiples, copy=copy)
    #        # We don't need to copy again
    #        copy = False

    #     # Recompute 'm' since it might change from dropping observations or from re-collapsing
    #     frame = frame.gen_m(force=True, copy=copy)

    #     frame.reset_index(drop=True, inplace=True)

    #     return frame
