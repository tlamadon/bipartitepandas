'''
Base class for bipartite networks in long or collapsed long form
'''
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

    def get_es(self):
        '''
        Return (collapsed) long form data reformatted into (collapsed) event study data.

        Returns:
            es_frame (BipartiteEventStudy(Collapsed)): BipartiteEventStudy(Collapsed) object generated from (collapsed) long data
        '''
        # Generate m column (the function checks if it already exists)
        self.gen_m()

        # Split workers by movers and stayers
        stayers = pd.DataFrame(self[self['m'] == 0])
        movers = pd.DataFrame(self[self['m'] == 1])
        self.logger.info('workers split by movers and stayers')

        # Add lagged values
        all_cols = self.included_cols()
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

        movers = movers[movers['i1'] == movers['i2']] # Ensure lagged values are for the same worker

        # Correct datatypes (shifting adds nans which converts all columns into float, correct columns that should be int)
        for col in all_cols:
            if (self.col_dtype_dict[col] == 'int') and (col != 'm'):
                for subcol in bpd.to_list(self.reference_dict[col]):
                    subcol_number = subcol.strip(col) # E.g. j1 will give 1
                    movers[col + '1' + subcol_number] = movers[col + '1' + subcol_number].astype(int)

        # Correct i
        movers.drop('i1', axis=1, inplace=True)
        movers.rename({'i2': 'i'}, axis=1, inplace=True)
        stayers.drop('i2', axis=1, inplace=True)
        stayers.rename({'i1': 'i'}, axis=1, inplace=True)

        # Keep only relevant columns
        stayers = stayers[keep_cols]
        movers = movers[keep_cols]
        self.logger.info('columns updated')

        # Merge stayers and movers
        data_es = pd.concat([stayers, movers]).reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(data_es.columns, key=bpd.col_order)
        data_es = data_es[sorted_cols]

        self.logger.info('data reformatted as event study')

        es_frame = self._constructor_es(data_es)
        es_frame.set_attributes(self, no_dict=True)

        return es_frame
