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
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, columns_req=[], columns_opt=[], reference_dict={}, col_dtype_dict={}, col_dict=None, **kwargs):
        if 't' not in columns_req:
            columns_req = ['t'] + columns_req
        reference_dict = bpd.update_dict({'j': 'j', 'y': 'y', 'g': 'g'}, reference_dict)
        # Initialize DataFrame
        super().__init__(*args, columns_req=columns_req, columns_opt=columns_opt, reference_dict=reference_dict, col_dtype_dict=col_dtype_dict, col_dict=col_dict, **kwargs)

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


        # Determine whether m, cluster columns exist
        clustered = self.col_included('j')

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
                if subcol not in ['m', 't1', 't2']: # Don't want lagged m
                    # Movers
                    movers[subcol + '1'] = movers[subcol].shift(periods=1) # Lagged value
                    movers.rename({subcol: subcol + '2'}, axis=1, inplace=True)
                    # Stayers (no lags)
                    stayers[subcol + '1'] = stayers[subcol]
                    stayers.rename({subcol: subcol + '2'}, axis=1, inplace=True)
                    if subcol != 'i': # Columns to keep
                        keep_cols += [subcol + '1', subcol + '2']
                elif subcol in ['t1', 't2']: # Treat t1 and t2 differently
                    if subcol == 't1':
                        one = 't11'
                        two = 't21'
                    else:
                        one = 't12'
                        two = 't22'
                    # Movers
                    movers[one] = movers[subcol].shift(periods=1) # Lagged value
                    movers.rename({subcol: two}, axis=1, inplace=True)
                    # Stayers (no lags)
                    stayers[one] = stayers[subcol]
                    stayers.rename({subcol: two}, axis=1, inplace=True)
                    # Columns to keep
                    keep_cols += [one, two]
                else:
                    keep_cols.append('m')

        movers = movers[movers['i1'] == movers['i2']] # Ensure lagged values are for the same worker

        # Correct datatypes (shifting adds nans which converts all columns into float, correct columns that should be int)
        for col in all_cols:
            if (self.col_dtype_dict[col] == 'int') and (col != 'm'):
                for subcol in bpd.to_list(self.reference_dict[col]):
                    movers[subcol + '1'] = movers[subcol + '1'].astype(int)
                    movers[subcol + '2'] = movers[subcol + '2'].astype(int) # FIXME need 2 as well because of issues with t

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
