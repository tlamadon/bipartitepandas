'''
Base class for bipartite networks in long or collapsed long form
'''
from xml.dom.minidom import Attr
import numpy as np
import pandas as pd
import bipartitepandas as bpd
from igraph import Graph
from collections import Counter

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

        # self.log('BipartiteLongBase object initialized', level='info')

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
            i_prev = bpd.fast_shift(i_col, 1, fill_value=-2)
            i_next = bpd.fast_shift(i_col, -1, fill_value=-2)
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
            frame.log("'m' column already included. Returning unaltered frame.", level='info')

        return frame

    def get_es(self, move_to_worker=False, is_sorted=False, copy=True):
        '''
        Return (collapsed) long form data reformatted into (collapsed) event study data.

        Arguments:
            move_to_worker (bool): if True, each move is treated as a new worker
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            es_frame (BipartiteEventStudy(Collapsed)): BipartiteEventStudy(Collapsed) object generated from (collapsed) long data
        '''
        # Sort data by i (and t, if included)
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)

        # Split workers by movers and stayers
        stayers = pd.DataFrame(frame.loc[frame.loc[:, 'm'].to_numpy() == 0, :])
        movers = pd.DataFrame(frame.loc[frame.groupby('i', sort=False)['m'].transform('max').to_numpy() > 0, :])
        frame.log('workers split by movers and stayers', level='info')

        # Add lagged values
        all_cols = frame._included_cols()

        # Columns to keep
        keep_cols = ['i']
        for col in all_cols:
            for subcol in bpd.to_list(frame.reference_dict[col]):
                # Get column number, e.g. j1 will give 1
                subcol_number = subcol.strip(col)
                ## Movers
                # Useful for t1 and t2: t1 should go to t11 and t21; t2 should go to t12 and t22
                plus_1 = col + '1' + subcol_number
                plus_2 = col + '2' + subcol_number
                # Lagged value
                movers.loc[:, plus_1] = np.roll(movers.loc[:, subcol].to_numpy(), 1)
                movers.rename({subcol: plus_2}, axis=1, inplace=True)

                if subcol not in ['i', 'm']:
                    ## Stayers (no lags)
                    stayers.loc[:, plus_1] = stayers.loc[:, subcol]
                    stayers.rename({subcol: plus_2}, axis=1, inplace=True)

                    # Columns to keep
                    keep_cols += [plus_1, plus_2]
                elif subcol == 'm':
                    # Columns to keep
                    keep_cols += ['m']

        # Ensure lagged values are for the same worker, and that neither observation is a stay (this ensures that if there is a mover who stays at a firm for multiple periods, e.g. A -> B -> B -> B -> C, then the event study will be A -> B, B -> C, with the middle B listed as a stayer)
        # Correct i because first row rolls over
        same_i = (movers.loc[:, 'i1'].to_numpy() == movers.loc[:, 'i2'].to_numpy())
        same_i[0] = False
        movers = movers.loc[same_i & (movers.loc[:, 'j1'].to_numpy() != movers.loc[:, 'j2'].to_numpy()), :]
        del same_i

        # Set 'm' = 1 for movers
        movers.drop(['m1', 'm2'], axis=1, inplace=True)
        movers.loc[:, 'm'] = 1

        # Correct datatypes (shifting adds nans which converts all columns into float, correct columns that should be int)
        for col in all_cols:
            if (frame.col_dtype_dict[col] == 'int') and (col != 'm'):
                for subcol in bpd.to_list(frame.reference_dict[col]):
                    # Get column number, e.g. j1 will give 1
                    subcol_number = subcol.strip(col)
                    movers.loc[:, col + '1' + subcol_number] = movers.loc[:, col + '1' + subcol_number].astype(int, copy=False)

        # Correct i
        movers.drop('i2', axis=1, inplace=True)
        movers.rename({'i1': 'i'}, axis=1, inplace=True)

        # Keep only relevant columns
        stayers = stayers.reindex(keep_cols, axis=1, copy=False)
        movers = movers.reindex(keep_cols, axis=1, copy=False)
        frame.log('columns updated', level='info')

        # Merge stayers and movers
        data_es = pd.concat([stayers, movers], ignore_index=True) # .reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(data_es.columns, key=bpd.col_order)
        data_es = data_es.reindex(sorted_cols, axis=1, copy=False)

        frame.log('data reformatted as event study', level='info')

        es_frame = frame._constructor_es(data_es, log=frame._log_on_indicator)
        es_frame._set_attributes(frame, no_dict=True)

        # Sort data by i and t
        es_frame = es_frame.sort_rows(is_sorted=False, copy=False)

        if move_to_worker:
            es_frame.loc[:, 'i'] = es_frame.index

        return es_frame

    def _get_spell_ids(self, is_sorted=False, copy=True):
        '''
        Generate array of spell ids, where a spell is defined as an uninterrupted period of time where a worker works at the same firm.

        Arguments:
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            copy (bool): if False, avoid copy

        Returns:
            spell_ids (NumPy Array): spell ids
        '''
        self.log('preparing to compute spell ids', level='info')

        # Sort data by i (and t, if included)
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)
        self.log('data sorted by i (and t, if included)', level='info')

        # Introduce lagged i and j
        i_col = frame.loc[:, 'i'].to_numpy()
        j_col = frame.loc[:, 'j'].to_numpy()
        i_prev = bpd.fast_shift(i_col, 1, fill_value=-2)
        j_prev = np.roll(j_col, 1)
        self.log('lagged i and j introduced', level='info')

        # Generate spell ids (allow for i != i_prev to ensure that consecutive workers at the same firm get counted as different spells)
        # Source: https://stackoverflow.com/questions/59778744/pandas-grouping-and-aggregating-consecutive-rows-with-same-value-in-column
        new_spell = (j_col != j_prev) | (i_col != i_prev)
        del i_col, j_col, i_prev, j_prev

        spell_ids = new_spell.cumsum()
        self.log('spell ids generated', level='info')

        return spell_ids

    def _drop_returns(self, how=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Drop observations where workers leave a firm then return to it.

        Arguments:
            how (str): if 'returns', drop observations where workers leave a firm then return to it; if 'returners', drop workers who ever leave then return to a firm
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteLongBase): dataframe that drops observations where workers leave a firm then return to it
        '''
        self.log('preparing to drop returns', level='info')

        # Sort data by i (and t, if included)
        frame = self.sort_rows(is_sorted=is_sorted, copy=copy)
        self.log('data sorted by i (and t, if included)', level='info')

        # Generate spell ids
        frame.loc[:, 'spell_ids'] = frame._get_spell_ids(is_sorted=True, copy=False)

        # Find returns
        frame.loc[:, 'return_row'] = (frame.groupby(['i', 'j'], sort=False)['spell_ids'].transform('nunique') > 1).astype(int, copy=False)

        # Check whether there are already no returns, or if we aren't dropping returns
        no_returns = (frame.loc[:, 'return_row'].sum() == 0)
        if no_returns or (not how):
            # Set frame.no_returns
            frame.no_returns = no_returns

            # Drop columns
            frame = frame.drop(['spell_ids', 'return_row'], axis=1, inplace=True)

            return frame
        del no_returns

        if how == 'returners':
            # Find returners
            frame.loc[:, 'return_row'] = frame.groupby('i', sort=False)['return_row'].transform('max')

        ## Drop returns
        return_rows = np.where(frame.loc[:, 'return_row'].to_numpy() == 1)[0]

        # Drop columns
        frame = frame.drop(['spell_ids', 'return_row'], axis=1, inplace=True)

        # Drop returns
        frame = frame.drop_rows(return_rows, drop_returns_to_stays=False, is_sorted=True, reset_index=reset_index, copy=False)

        # Set frame.no_returns
        frame.no_returns = True

        self.log('returns dropped', level='info')

        return frame

    def _prep_cluster(self, stayers_movers=None, t=None, weighted=True, is_sorted=False, copy=False):
        '''
        Prepare data for clustering.

        Arguments:
            stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers
            t (int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)
            weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)
            is_sorted (bool): used for event study format, does nothing for long
            copy (bool): if False, avoid copy
        Returns:
            data (Pandas DataFrame): data prepared for clustering
            weights (NumPy Array or None): if weighted=True, gives NumPy array of firm weights for clustering; otherwise, is None
            jids (NumPy Array): firm ids of firms in subset of data used to cluster
        '''
        if copy:
            frame = self.copy()
        else:
            frame = self

        if stayers_movers is not None:
            if stayers_movers == 'stayers':
                frame = frame.loc[frame.loc[:, 'm'].to_numpy() == 0, :]
            elif stayers_movers == 'movers':
                frame = frame.loc[frame.loc[:, 'm'].to_numpy() > 0, :]
            else:
                raise NotImplementedError("Invalid 'stayers_movers' option, {}. Valid options are 'stayers', 'movers', or None.".format(stayers_movers))

        # If period-level, then only use data for that particular period
        if t is not None:
            if isinstance(frame, bpd.BipartiteLong):
                frame = frame.loc[frame.loc[:, 't'].to_numpy() == t, :]
            else:
                raise NotImplementedError("Cannot use data from a particular period with collapsed data. Data can be converted to long format using the '.uncollapse()' method.")

        # Create weights
        ##### Disable Pandas warning #####
        pd.options.mode.chained_assignment = None
        if weighted:
            if frame._col_included('w'):
                frame.loc[:, 'row_weights'] = frame.loc[:, 'w']
                weights = frame.groupby('j')['w'].sum().to_numpy()
            else:
                frame.loc[:, 'row_weights'] = 1
                weights = frame.groupby('j').size().to_numpy()
        else:
            frame.loc[:, 'row_weights'] = 1
            weights = None
        ##### Re-enable Pandas warning #####
        pd.options.mode.chained_assignment = 'warn'

        # Get unique firm ids (must sort)
        jids = np.sort(frame.loc[:, 'j'].unique())

        return frame, weights, jids

    def _leave_one_observation_out(self, cc_list, max_j, component_size_variable='firms', drop_returns_to_stays=False, frame_largest_cc=None, is_sorted=False):
        '''
        Extract largest leave-one-observation-out connected component.

        Arguments:
            cc_list (list of lists): each entry is a connected component
            max_j (int): maximum j
            component_size_variable (str): how to determine largest leave-one-observation-out connected component. Options are 'len'/'length' (length of frame), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), and 'movers' (number of unique movers)
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            frame_largest_cc (BipartiteLongBase): dataframe of baseline largest leave-one-observation-out connected component
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.

        Returns:
            frame_largest_cc (BipartiteLongBase): dataframe of largest leave-one-observation-out connected component
        '''
        # Sort data by i (and t, if included)
        self.sort_rows(is_sorted=is_sorted, copy=False)

        for cc in sorted(cc_list, reverse=True, key=len):
            cc = np.array(cc)
            cc_j = cc[cc <= max_j]
            if (frame_largest_cc is not None) and (component_size_variable == 'firms'):
                # If looking at number of firms, can check if frame_cc is already smaller than frame_largest_cc before any computations
                try:
                    skip = (frame_largest_cc.comp_size >= len(cc_j))
                except AttributeError:
                    frame_largest_cc.comp_size = frame_largest_cc.n_firms()
                    skip = (frame_largest_cc.comp_size >= len(cc_j))

                if skip:
                    continue

            # Keep observations in connected components (NOTE: this does not require a copy)
            frame_cc = self.keep_ids('j', cc_j, drop_returns_to_stays, is_sorted=True, copy=False)

            if frame_largest_cc is not None:
                # If frame_cc is already smaller than frame_largest_cc
                skip = bpd.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='geq')

                if skip:
                    continue

            # Remove firms with only 1 mover observation (can have 1 mover with multiple observations)
            frame_cc = frame_cc.min_moves_frame(2, drop_returns_to_stays, is_sorted=True, copy=False)

            if frame_largest_cc is not None:
                # If frame_cc is already smaller than frame_largest_cc
                skip = bpd.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='geq')

                if skip:
                    continue

            # Construct graph
            G2, max_j2 = frame_cc._construct_graph('leave_one_observation_out', is_sorted=True)

            # Extract articulation rows
            articulation_rows = frame_cc._get_articulation_obs(G2, max_j2, is_sorted=True)

            if len(articulation_rows) > 0:
                # If new frame is not leave-one-out connected, recompute connected components after dropping articulation rows (but note that articulation rows should be kept in the final dataframe) (NOTE: this does not require a copy)
                G2, max_j2 = frame_cc.drop_rows(articulation_rows, drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False)._construct_graph('leave_one_observation_out', is_sorted=True)
                cc_list_2 = G2.components()
                # Recursion step
                frame_cc = frame_cc._leave_one_observation_out(cc_list=cc_list_2, max_j=max_j2, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays, frame_largest_cc=frame_largest_cc, is_sorted=True)

            if frame_largest_cc is None:
                # If in the first round
                replace = True
            elif frame_cc is None:
                # If the biconnected components have recursively been eliminated
                replace = False
            else:
                replace = bpd.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='lt')
            if replace:
                frame_largest_cc = frame_cc
                try:
                    # Reset comp_size attribute
                    del frame_largest_cc.comp_size
                except AttributeError:
                    pass

        # Remove comp_size attribute before return
        try:
            del frame_largest_cc.comp_size
        except AttributeError:
            pass

        # Return largest leave-one-observation-out component
        return frame_largest_cc

    # def _leave_one_observation_out(self, cc_list, component_size_variable='firms', drop_returns_to_stays=False, frame_largest_cc=None):
    #     ''' # FIXME this is orders of magnitude slower than the new implementation, on sample data it looks at 275x more observations to check whether the set is leave-one-out connected (it also results in a smaller connected set)
    #     Extract largest leave-one-observation-out connected component.

    #     Arguments:
    #         cc_list (list of lists): each entry is a connected component
    #         component_size_variable (str): how to determine largest leave-one-observation-out connected component. Options are 'len'/'length' (length of frame), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), and 'movers' (number of unique movers)
    #         drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
    #         frame_largest_cc (BipartiteLongBase): dataframe of baseline largest leave-one-observation-out connected component

    #     Returns:
    #         frame_largest_cc (BipartiteLongBase): dataframe of largest leave-one-observation-out connected component
    #     '''
    #     for cc in sorted(cc_list, reverse=True, key=len):
    #         if (frame_largest_cc is not None) and (component_size_variable == 'firms'):
    #             # If looking at number of firms, can check if frame_cc is already smaller than frame_largest_cc before any computations
    #             try:
    #                 skip = (frame_largest_cc.comp_size >= len(cc))
    #             except AttributeError:
    #                 frame_largest_cc.comp_size = frame_largest_cc.n_firms()
    #                 skip = (frame_largest_cc.comp_size >= len(cc))

    #             if skip:
    #                 continue

    #         # Keep observations in connected components (NOTE: this does not require a copy)
    #         frame_cc = self.keep_ids('j', cc, drop_returns_to_stays, is_sorted=True, copy=False)

    #         if frame_largest_cc is not None:
    #             # If frame_cc is already smaller than frame_largest_cc
    #             skip = bpd.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='geq')

    #             if skip:
    #                 continue

    #         # Remove firms with only 1 mover observation (can have 1 mover with multiple observations)
    #         frame_cc = frame_cc.min_moves_frame(2, drop_returns_to_stays, is_sorted=True, copy=False)

    #         if frame_largest_cc is not None:
    #             # If frame_cc is already smaller than frame_largest_cc
    #             skip = bpd.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='geq')

    #             if skip:
    #                 continue

    #         # Construct graph
    #         G2 = frame_cc._construct_graph('leave_one_observation_out')

    #         # Extract articulation firms
    #         articulation_firms = G2.articulation_points()

    #         if len(articulation_firms) > 0:
    #             # If there are articulation firms
    #             # Extract articulation rows
    #             print('n rows:', len(frame_cc.loc[(frame_cc.loc[:, 'j'].isin(articulation_firms)) & (frame_cc.loc[:, 'm'].to_numpy() > 0), :].index.to_numpy()))
    #             articulation_rows = frame_cc._get_articulation_obs(G2, frame_cc.loc[(frame_cc.loc[:, 'j'].isin(articulation_firms)) & (frame_cc.loc[:, 'm'].to_numpy() > 0), :].index.to_numpy())

    #             if len(articulation_rows) > 0:
    #                 print('n articulation rows:', len(articulation_rows))
    #                 # If new frame is not leave-one-out connected, recompute connected components after dropping articulation rows (but note that articulation rows should be kept in the final dataframe) (NOTE: this does not require a copy)
    #                 G2 = frame_cc.drop_rows(articulation_rows, drop_returns_to_stays, is_sorted=True, reset_index=False, copy=False)._construct_graph('leave_one_observation_out')
    #                 cc_list_2 = G2.components()
    #                 # Recursion step
    #                 frame_cc = frame_cc._leave_one_observation_out(cc_list=cc_list_2, component_size_variable=component_size_variable, drop_returns_to_stays=drop_returns_to_stays, frame_largest_cc=frame_largest_cc)

    #         if frame_largest_cc is None:
    #             # If in the first round
    #             replace = True
    #         elif frame_cc is None:
    #             # If the biconnected components have recursively been eliminated
    #             replace = False
    #         else:
    #             replace = bpd.compare_frames(frame_largest_cc, frame_cc, size_variable=component_size_variable, operator='lt')
    #         if replace:
    #             frame_largest_cc = frame_cc
    #             try:
    #                 # Reset comp_size attribute
    #                 del frame_largest_cc.comp_size
    #             except AttributeError:
    #                 pass

    #     try:
    #         # Remove comp_size attribute
    #         del frame_largest_cc.comp_size
    #     except AttributeError:
    #         pass

    #     # Return largest leave-one-observation-out component
    #     return frame_largest_cc

    def _leave_one_firm_out(self, bcc_list, component_size_variable='firms', drop_returns_to_stays=False, is_sorted=False):
        '''
        Extract largest leave-one-firm-out connected component.

        Arguments:
            bcc_list (list of lists): each entry is a biconnected component
            component_size_variable (str): how to determine largest leave-one-firm-out connected component. Options are 'len'/'length' (length of frame), 'firms' (number of unique firms), 'workers' (number of unique workers), 'stayers' (number of unique stayers), and 'movers' (number of unique movers)
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.

        Returns:
            frame_largest_bcc (BipartiteLongBase): dataframe of largest leave-one-out connected component
        '''
        # Sort data by i (and t, if included)
        self.sort_rows(is_sorted=is_sorted, copy=False)

        # This will become the largest leave-one-firm-out component
        frame_largest_bcc = None

        for bcc in sorted(bcc_list, reverse=True, key=len):
            if (frame_largest_bcc is not None) and (component_size_variable == 'firms'):
                # If looking at number of firms, can check if frame_cc is already smaller than frame_largest_cc before any computations
                try:
                    skip = (frame_largest_bcc.comp_size >= len(bcc))
                except AttributeError:
                    frame_largest_bcc.comp_size = frame_largest_bcc.n_firms()
                    skip = (frame_largest_bcc.comp_size >= len(bcc))

                if skip:
                    continue

            # Keep observations in biconnected components (NOTE: this does not require a copy)
            frame_bcc = self.keep_ids('j', bcc, drop_returns_to_stays, is_sorted=True, copy=False)

            if frame_largest_bcc is not None:
                # If frame_bcc is already smaller than frame_largest_bcc
                skip = bpd.compare_frames(frame_largest_bcc, frame_bcc, size_variable=component_size_variable, operator='geq')

                if skip:
                    continue

            # Remove firms with only 1 mover observation (can have 1 mover with multiple observations)
            # This fixes a discrepency between igraph's biconnected components and the definition of leave-one-out connected set, where biconnected components is True if a firm has only 1 mover, since then it disappears from the graph - but leave-one-out requires the set of firms to remain unchanged
            frame_bcc = frame_bcc.min_moves_frame(2, drop_returns_to_stays, is_sorted=True, copy=False)

            if frame_largest_bcc is not None:
                # If frame_bcc is already smaller than frame_largest_bcc
                skip = bpd.compare_frames(frame_largest_bcc, frame_bcc, size_variable=component_size_variable, operator='geq')

                if skip:
                    continue

            # # Recompute biconnected components
            # G2 = frame_bcc._construct_biconnected_graph()
            # bcc_list_2 = G2.biconnected_components()

            # # If new frame is not biconnected after dropping firms with 1 mover observation, recompute biconnected components
            # if not ((len(bcc_list_2) == 1) and (len(bcc_list_2[0]) == frame_bcc.n_firms())):
            #     frame_bcc = frame_bcc._leave_one_out(bcc_list_2, component_size_variable, drop_returns_to_stays)

            if frame_largest_bcc is None:
                # If in the first round
                replace = True
            elif frame_bcc is None:
                # If the biconnected components have recursively been eliminated
                replace = False
            else:
                replace = bpd.compare_frames(frame_largest_bcc, frame_bcc, size_variable=component_size_variable, operator='lt')
            if replace:
                frame_largest_bcc = frame_bcc
                try:
                    # Reset comp_size attribute
                    del frame_largest_bcc.comp_size
                except AttributeError:
                    pass

        try:
            # Remove comp_size attribute
            del frame_largest_bcc.comp_size
        except AttributeError:
            pass

        # Return largest biconnected component
        return frame_largest_bcc

    def _construct_firm_linkages(self, is_sorted=False):
        '''
        Construct numpy array linking firms by movers, for use with connected components.

        Arguments:
            is_sorted (bool): used for _construct_firm_worker_linkages, does nothing for _construct_firm_linkages

        Returns:
            (tuple of NumPy Array, int): (firm linkages, maximum firm id)
        '''
        move_rows = (self.loc[:, 'm'].to_numpy() > 0)
        i_col = self.loc[move_rows, 'i'].to_numpy()
        j_col = self.loc[move_rows, 'j'].to_numpy()
        j_next = np.roll(j_col, -1)
        i_match = (i_col == bpd.fast_shift(i_col, -1, fill_value=-2))
        j_col = j_col[i_match]
        j_next = j_next[i_match]
        linkages = np.stack([j_col, j_next], axis=1)
        max_j = np.max(linkages)

        return linkages, max_j

    def _construct_firm_double_linkages(self, is_sorted=False):
        '''
        Construct numpy array linking firms by movers, for use with leave-one-firm-out components.

        Arguments:
            is_sorted (bool): used for _construct_firm_worker_linkages, does nothing for _construct_firm_double_linkages

        Returns:
            (tuple of NumPy Array, int): (firm linkages, maximum firm id)
        '''
        move_rows = (self.loc[:, 'm'].to_numpy() > 0)
        i_col = self.loc[move_rows, 'i'].to_numpy()
        j_col = self.loc[move_rows, 'j'].to_numpy()
        i_next = bpd.fast_shift(i_col, -1, fill_value=-2)
        j_next = np.roll(j_col, -1)
        valid_next = (i_col == i_next)
        base_linkages = np.stack([j_col[valid_next], j_next[valid_next]], axis=1)
        i_next_2 = bpd.fast_shift(i_col, -2, fill_value=-2)
        j_next_2 = np.roll(j_col, -2)
        valid_next_2 = (i_col == i_next_2)
        secondary_linkages = np.stack([j_col[valid_next_2], j_next_2[valid_next_2]], axis=1)
        linkages = np.concatenate([base_linkages, secondary_linkages], axis=0)
        max_j = np.max(linkages)

        return linkages, max_j

    # def _biconnected_linkages_indices(self):
    #     '''
    #     Construct numpy array of original indices for biconnected linkages. The first column tells you, for each link in the graph, what index the first observation in the link is coming from; and the second column tells you, for each link in the graph, what index the second observation in the link is coming from.

    #     Returns:
    #         (NumPy Array): original indices
    #     '''
    #     move_rows = (self.loc[:, 'm'].to_numpy() > 0)
    #     i_col = self.loc[move_rows, 'i'].to_numpy()
    #     indices = self.loc[move_rows, :].index.to_numpy()
    #     i_next = bpd.fast_shift(i_col, -1, fill_value=-2)
    #     indices_next = np.roll(indices, -1)
    #     valid_next = (i_col == i_next)
    #     base_indices = np.stack([indices[valid_next], indices_next[valid_next]], axis=1)
    #     i_next_2 = bpd.fast_shift(i_col, -2, fill_value=-2)
    #     indices_next_2 = np.roll(indices, -2)
    #     valid_next_2 = (i_col == i_next_2)
    #     secondary_indices = np.stack([indices[valid_next_2], indices_next_2[valid_next_2]], axis=1)
    #     original_indices = np.concatenate([base_indices, secondary_indices], axis=0)
    #     return original_indices

    # def _get_articulation_obs(self, G, obs_list):
    #     ''' # FIXME this is for the old leave-one-out method
    #     Compute articulation observations for self, by checking whether self is leave-one-observation-out connected when dropping selected observations one at a time.

    #     Arguments:
    #         G (igraph Graph): graph linking firms by movers
    #         obs_list (list): list of observations to drop

    #     Returns:
    #         (list): articulation observations for self
    #     '''
    #     # Get original indices for biconnected linkages
    #     original_indices = self._biconnected_linkages_indices()
    #     index_first = original_indices[:, 0]
    #     index_second = original_indices[:, 1]

    #     # Save articulation observations (observations that disconnect the graph when they are removed)
    #     articulation_obs = []

    #     # Check if each observation is an articulation observation
    #     for obs in obs_list:
    #         G_obs = G.copy()
    #         # Observation gives an index in the frame, but we need an index for the graph
    #         try:
    #             # If observation is first in pair
    #             obs_indices = list(np.where(index_first == obs)[0])
    #         except IndexError:
    #             # If observation isn't first in pair
    #             obs_indices = []
    #         try:
    #             # If observation is second in pair
    #             obs_indices += list(np.where(index_second == obs)[0])
    #         except IndexError:
    #             # If observation isn't second in pair
    #             pass

    #         # Delete row(s)
    #         # print(G_row.es())
    #         G_obs.delete_edges(obs_indices)

    #         # Check whether removing row(s) disconnects graph
    #         if not G_obs.is_connected():
    #             articulation_obs += [obs]

    #     return articulation_obs

    # def _get_articulation_obs(self, G, obs_list):
    #     ''' # FIXME this is around twice as slow as other implementation
    #     Compute articulation observations for self, by checking whether self is leave-one-observation-out connected when dropping selected observations one at a time.

    #     Arguments:
    #         G (igraph Graph): graph linking firms by movers
    #         obs_list (list): list of observations to drop

    #     Returns:
    #         (list): articulation observations for self
    #     '''
    #     # Get original indices for biconnected linkages
    #     original_indices = self._biconnected_linkages_indices()
    #     index_first = original_indices[:, 0]
    #     index_second = original_indices[:, 1]

    #     # Save articulation observations (observations that disconnect the graph when they are removed)
    #     articulation_obs = []

    #     # Check if each observation is an articulation observation
    #     for obs in obs_list:
    #         # Observation gives an index in the frame, but we need an index for the graph
    #         try:
    #             # If observation is first in pair
    #             obs_indices = list(np.where(index_first == obs)[0])
    #         except IndexError:
    #             # If observation isn't first in pair
    #             obs_indices = []
    #         try:
    #             # If observation is second in pair
    #             obs_indices += list(np.where(index_second == obs)[0])
    #         except IndexError:
    #             # If observation isn't second in pair
    #             pass

    #         # Shift indices to account for dropping and re-adding rows to graph
    #         original_indices = np.concatenate([original_indices[np.delete(np.arange(len(original_indices)), obs_indices), :], original_indices[obs_indices, :]])
    #         index_first = original_indices[:, 0]
    #         index_second = original_indices[:, 1]

    #         # Save graph tuples of observations to be removed, so we can add them back later
    #         obs_tuples = [G.es()[obs_index].tuple for obs_index in obs_indices]

    #         # Delete row(s)
    #         G.delete_edges(obs_indices)

    #         # Check whether removing row(s) disconnects graph
    #         if not G.is_connected():
    #             articulation_obs += [obs]

    #         # Add rows back
    #         G.add_edges(obs_tuples)

    #     return articulation_obs

    def keep_ids(self, id_col, keep_ids_list, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Only keep ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            keep_ids_list (list): ids to keep
            drop_returns_to_stays (bool): used only if id_col is 'j' or 'g' and using BipartiteLongCollapsed format. If True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer).
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteLongBase): dataframe with ids in the given set
        '''
        keep_ids_list = set(keep_ids_list)
        if len(keep_ids_list) == self.n_unique_ids(id_col):
            # If keeping everything
            if copy:
                return self.copy()
            return self

        frame = self.loc[self.loc[:, id_col].isin(keep_ids_list), :]

        if id_col in ['j', 'g']:
            if isinstance(frame, bpd.BipartiteLongCollapsed):
                # If BipartiteLongCollapsed
                frame = frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, copy=copy)
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

    def drop_ids(self, id_col, drop_ids_list, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Drop ids belonging to a given set of ids.

        Arguments:
            id_col (str): column of ids to consider ('i', 'j', or 'g')
            drop_ids_list (list): ids to drop
            drop_returns_to_stays (bool): used only if id_col is 'j' or 'g' and using BipartiteLongCollapsed format. If True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer).
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteLongBase): dataframe with ids outside the given set
        '''
        drop_ids_list = set(drop_ids_list)
        if len(drop_ids_list) == 0:
            # If nothing input
            if copy:
                return self.copy()
            return self

        frame = self.loc[~(self.loc[:, id_col].isin(drop_ids_list)), :]

        if id_col in ['j', 'g']:
            if isinstance(frame, bpd.BipartiteLongCollapsed):
                # If BipartiteLongCollapsed
                frame = frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, copy=copy)
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

    def keep_rows(self, rows_list, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Only keep particular rows.

        Arguments:
            rows_list (list): rows to keep
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            frame (BipartiteLongBase): dataframe with given rows
        '''
        rows_list = set(rows_list)
        if len(rows_list) == len(self):
            # If keeping everything
            if copy:
                return self.copy()
            return self
        rows_list = sorted(list(rows_list))

        frame = self.iloc[rows_list]

        if isinstance(frame, bpd.BipartiteLongCollapsed):
            # If BipartiteLongCollapsed
            frame = frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, copy=copy)
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

        n_obs = self.loc[:, 'j'].value_counts(sort=False)
        valid_firms = n_obs[n_obs.to_numpy() >= threshold].index.to_numpy()

        return valid_firms

    @bpd.recollapse_loop(False)
    def min_obs_frame(self, threshold=2, drop_returns_to_stays=False, is_sorted=False, copy=True):
        '''
        Keep firms with at least `threshold` many observations.

        Arguments:
            threshold (int): minimum number of observations required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
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
            frame = frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, copy=copy)
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

    @bpd.recollapse_loop(False)
    def min_workers_frame(self, threshold=15, drop_returns_to_stays=False, is_sorted=False, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many workers.

        Arguments:
            threshold (int): minimum number of workers required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
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
            frame = frame.recollapse(drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, copy=copy)
            # We don't need to copy again
            copy = False

        # Recompute 'm' since it might change from dropping observations or from re-collapsing
        frame = frame.gen_m(force=True, copy=copy)

        frame.reset_index(drop=True, inplace=True)

        return frame

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

        return self.loc[self.loc[:, 'm'].to_numpy() > 0].min_obs_firms(threshold=threshold)

    @bpd.recollapse_loop(True)
    def min_moves_frame(self, threshold=2, drop_returns_to_stays=False, is_sorted=False, reset_index=True, copy=True):
        '''
        Return dataframe of firms with at least `threshold` many moves. Note that a single mover can have multiple moves at the same firm.

        Arguments:
            threshold (int): minimum number of moves required to keep a firm
            drop_returns_to_stays (bool): if True, when recollapsing collapsed data, drop observations that need to be recollapsed instead of collapsing (this is for computational efficiency when re-collapsing data for leave-one-out connected components, where intermediate observations can be dropped, causing a worker who returns to a firm to become a stayer)
            is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Set to True if already sorted.
            reset_index (bool): if True, reset index at end
            copy (bool): if False, avoid copy

        Returns:
            (BipartiteBase): dataframe of firms with sufficiently many moves
        '''
        if threshold == 0:
            # If no threshold
            if copy:
                return self.copy()
            return self

        valid_firms = self.min_moves_firms(threshold)

        return self.keep_ids('j', keep_ids_list=valid_firms, drop_returns_to_stays=drop_returns_to_stays, is_sorted=is_sorted, reset_index=reset_index, copy=copy)
