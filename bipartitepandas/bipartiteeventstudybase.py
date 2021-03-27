'''
Base class for bipartite networks in event study or collapsed event study form
'''
import pandas as pd
import bipartitepandas as bpd

class BipartiteEventStudyBase(bpd.BipartiteBase):
    '''
    Base class for BipartiteEventStudy and BipartiteEventStudyCollapsed, where BipartiteEventStudy and BipartiteEventStudyCollapsed give a bipartite network of firms and workers in event study and collapsed event study form, respectively. Contains generalized methods. Inherits from BipartiteBase.

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
        columns_opt += ['year']
        reference_dict = bpd.update_dict({'fid': ['f1i', 'f2i'], 'comp': ['y1', 'y2'], 'j': ['j1', 'j2']}, reference_dict)
        # Initialize DataFrame
        super().__init__(*args, columns_req=columns_req, columns_opt=columns_opt, reference_dict=reference_dict, col_dtype_dict=col_dtype_dict, col_dict=col_dict, **kwargs)

        # self.logger.info('BipartiteEventStudyBase object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteEventStudyBase

    def clean_data(self):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Returns:
            frame (BipartiteEventStudyBase): BipartiteEventStudyBase with cleaned data
        '''
        frame = bpd.BipartiteBase.clean_data(self)

        frame.logger.info('beginning BipartiteEventStudyBase data cleaning')
        frame.logger.info('checking quality of data')
        frame.data_validity()

        frame.logger.info('BipartiteEventStudyBase data cleaning complete')

        return frame

    def data_validity(self, inplace=True):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteEventStudyBase): BipartiteEventStudyBase with corrected columns and attributes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        success_stayers = True
        success_movers = True

        stayers = frame[frame['m'] == 0]
        movers = frame[frame['m'] == 1]

        frame.logger.info('--- checking firms ---')
        firms_stayers = (stayers['f1i'] != stayers['f2i']).sum()
        firms_movers = (movers['f1i'] == movers['f2i']).sum()

        frame.logger.info('stayers with different firms (should be 0):' + str(firms_stayers))
        frame.logger.info('movers with same firm (should be 0):' + str(firms_movers))
        if firms_stayers > 0:
            success_stayers = False
        if firms_movers > 0:
            success_movers = False

        frame.logger.info('--- checking income ---')
        income_stayers = (stayers['y1'] != stayers['y2']).sum()

        frame.logger.info('stayers with different income (should be 0):' + str(income_stayers))
        if income_stayers > 0:
            success_stayers = False

        frame.logger.info('Overall success for stayers:' + str(success_stayers))
        frame.logger.info('Overall success for movers:' + str(success_movers))

        return frame

    def get_cs(self):
        '''
        Return event study data reformatted into cross section data.

        Returns:
            data_cs (Pandas DataFrame): cross section data
        '''
        # Determine whether m column exists
        if not self.col_included('m'):
            self.gen_m()

        sdata = pd.DataFrame(self[self['m'] == 0])
        jdata = pd.DataFrame(self[self['m'] == 1])

        # # Assign some values
        # ns = len(sdata)
        # nm = len(jdata)

        # # Reset index
        # sdata.set_index(np.arange(ns) + 1 + nm)
        # jdata.set_index(np.arange(nm) + 1)

        # Columns used for constructing cross section
        cs_cols = self.included_cols(flat=True)

        rename_dict = {
            'f1i': 'f2i',
            'f2i': 'f1i',
            'y1': 'y2',
            'y2': 'y1',
            'year_1': 'year_2',
            'year_2': 'year_1',
            'year_start_1': 'year_start_2',
            'year_start_2': 'year_start_1',
            'year_end_1': 'year_end_2',
            'year_end_2': 'year_end_1',
            'w1': 'w2',
            'w2': 'w1',
            'j1': 'j2',
            'j2': 'j1'
        }

        # Combine the 2 data-sets
        data_cs = pd.concat([
            sdata[cs_cols].assign(cs=1),
            jdata[cs_cols].assign(cs=1),
            jdata[cs_cols].rename(rename_dict, axis=1).assign(cs=0)
        ], ignore_index=True)

        # Sort columns
        sorted_cols = sorted(data_cs.columns, key=bpd.col_order)
        data_cs = data_cs[sorted_cols]

        self.logger.info('mover and stayer event study datasets combined into cross section')

        return data_cs
