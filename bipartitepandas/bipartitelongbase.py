'''
Base class for bipartite networks in long or collapsed long form
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import igraph as ig

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
        Generate m column for data (m == 0 if stayer, m == 1 if mover).

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
            frame['m'] = (frame.groupby('i')['j'].transform('nunique').to_numpy() > 1).astype(int)

            frame.col_dict['m'] = 'm'

            # Sort columns
            frame = frame.sort_cols(copy=False)

        else:
            self.logger.info("'m' column already included. Returning unaltered frame.")

        return frame

    def get_es(self):
        '''
        Return (collapsed) long form data reformatted into (collapsed) event study data.

        Returns:
            es_frame (BipartiteEventStudy(Collapsed)): BipartiteEventStudy(Collapsed) object generated from (collapsed) long data
        '''
        # Split workers by movers and stayers
        stayers = pd.DataFrame(self[self['m'].to_numpy() == 0])
        movers = pd.DataFrame(self[self['m'].to_numpy() == 1])
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
                    movers[plus_1] = movers[subcol].shift(periods=1) # Lagged value
                    movers.rename({subcol: plus_2}, axis=1, inplace=True)
                    # Stayers (no lags)
                    stayers[plus_1] = stayers[subcol]
                    stayers.rename({subcol: plus_2}, axis=1, inplace=True)
                    if subcol != 'i': # Columns to keep
                        keep_cols += [plus_1, plus_2]
                else:
                    keep_cols.append('m')

        # Ensure lagged values are for the same worker
        # NOTE: cannot use .to_numpy()
        movers = movers[movers['i1'] == movers['i2']]

        # Correct datatypes (shifting adds nans which converts all columns into float, correct columns that should be int)
        for col in all_cols:
            if (self.col_dtype_dict[col] == 'int') and (col != 'm'):
                for subcol in bpd.to_list(self.reference_dict[col]):
                    subcol_number = subcol.strip(col) # E.g. j1 will give 1
                    movers[col + '1' + subcol_number] = movers[col + '1' + subcol_number].astype(int)

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

        # Recompute 'm'
        es_frame = es_frame.gen_m(force=True, copy=False)

        return es_frame

    def _construct_graph(self):
        '''
        Construct igraph graph linking firms by movers.

        Returns:
            G (igraph Graph): igraph graph
        '''
        # Match firms linked by worker moves
        i_col = self['i'].to_numpy()
        j_col = self['j'].to_numpy()
        j_prev = np.roll(j_col, 1)
        i_match = (i_col == np.roll(i_col, 1))
        j_col = j_col[i_match]
        j_prev = j_prev[i_match]
        # # Source: https://stackoverflow.com/questions/16453465/multi-column-factorize-in-pandas
        # j_zip = pd._libs.lib.fast_zip([j_col, j_prev])
        j_zip = np.stack([j_col, j_prev], axis=1)
        G = ig.Graph(n=self.n_firms(), edges=j_zip)

        return G

    def leave_one_out(self, bcc_list, drop_multiples=False):
        '''
        Extract largest leave-one-out connected set.

        Arguments:
            bcc_list (list of lists): list of lists, where sublists give firms in each biconnected component
            drop_multiples (bool): if True, rather than collapsing over spells, drop any spells with multiple observations (this is for computational efficiency when re-collapsing data)

        Returns:
            frame_largest_bcc (BipartiteLongBase): largest leave-one-out connected set
        '''
        # This will become the largest biconnected component
        frame_largest_bcc = []

        for bcc in bcc_list:
            # Observations in biconnected components
            frame_bcc = self[self['j'].isin(bcc)]

            if isinstance(self, bpd.BipartiteLongCollapsed):
                frame_bcc = frame_bcc.recollapse(drop_multiples=drop_multiples, copy=False)

            # Recompute 'm' since it might change from dropping firms or from re-collapsing
            frame_bcc = frame_bcc.gen_m(force=True, copy=True)

            # Remove firms with only 1 mover observation (can have 1 mover with multiple observations)
            # This fixes a discrepency between igraph's biconnected components and the definition of leave-one-out connected set, where biconnected components is True if a firm has only 1 mover, since then it disappears from the graph - but leave-one-out requires the set of firms to remain unchanged
            frame_bcc = frame_bcc[frame_bcc.groupby('j')['m'].transform('sum') >= 2]

            # Recompute 'm' since it might change from dropping firms
            frame_bcc = frame_bcc.gen_m(force=True, copy=True)

            # Recompute biconnected components
            G2 = frame_bcc._construct_graph()
            bcc_list_2 = G2.biconnected_components()

            # If new frame is biconnected after dropping firms with 1 mover observation, return it
            if (len(bcc_list_2) == 1) and (len(bcc_list_2[0]) == frame_bcc.n_firms()):
                frame_largest_bcc = max(frame_largest_bcc, frame_bcc, key=len)
            # Otherwise, loop again
            else:
                frame_largest_bcc = max(frame_largest_bcc, frame_bcc.leave_one_out(bcc_list_2, drop_multiples), key=len)

        # Return largest biconnected component
        return frame_largest_bcc
