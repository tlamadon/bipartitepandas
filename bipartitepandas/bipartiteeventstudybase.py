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

    def clean_data(self, user_clean={}):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Arguments:
            user_clean (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids

        Returns:
            frame (BipartiteEventStudyBase): BipartiteEventStudyBase with cleaned data
        '''
        frame = bpd.BipartiteBase.clean_data(self, user_clean)

        frame.logger.info('beginning BipartiteEventStudyBase data cleaning')
        frame.logger.info('checking quality of data')
        frame = frame.data_validity()

        frame.logger.info('BipartiteEventStudyBase data cleaning complete')

        return frame

    def data_validity(self):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Returns:
            frame (BipartiteEventStudyBase): BipartiteEventStudyBase with corrected columns and attributes
        '''
        frame = self # .copy()

        success_stayers = True
        success_movers = True

        stayers = frame[frame['m'] == 0]
        movers = frame[frame['m'] == 1]

        frame.logger.info('--- checking firms ---')
        firms_stayers = (stayers['j1'] != stayers['j2']).sum()
        firms_movers = (movers['j1'] == movers['j2']).sum()

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
        Return (collapsed) event study data reformatted into cross section data.

        Returns:
            data_cs (Pandas DataFrame): cross section data
        '''
        # Generate m column (the function checks if it already exists)
        self.gen_m()

        sdata = pd.DataFrame(self[self['m'] == 0])
        jdata = pd.DataFrame(self[self['m'] == 1])

        # Columns used for constructing cross section
        cs_cols = self.included_cols(flat=True)

        # Dictionary to swap names for cs=0 (these rows contain period-2 data for movers, so must swap columns for all relevant information to be contained in the same column (e.g. must move y2 into y1, otherwise bottom rows are just duplicates))
        rename_dict = {}
        for col in self.included_cols():
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
            sdata[cs_cols].assign(cs=1),
            jdata[cs_cols].assign(cs=1),
            jdata[cs_cols].rename(rename_dict, axis=1).assign(cs=0)
        ], ignore_index=True)

        # Sort columns
        sorted_cols = sorted(data_cs.columns, key=bpd.col_order)
        data_cs = data_cs[sorted_cols]

        self.logger.info('mover and stayer event study datasets combined into cross section')

        return data_cs

    def get_long(self, return_df=False):
        '''
        Return (collapsed) event study data reformatted into (collapsed) long form.

        Arguments:
            return_df (bool): if True, return a Pandas dataframe instead of a BipartiteLong(Collapsed) dataframe

        Returns:
            long_frame (BipartiteLong(Collapsed) or Pandas DataFrame): BipartiteLong(Collapsed) or Pandas dataframe generated from (collapsed) event study data
        '''
        # Generate m column (the function checks if it already exists)
        self.gen_m()

        # Dictionary to swap names (necessary for last row of data, where period-2 observations are not located in subsequent period-1 column (as it doesn't exist), so must append the last row with swapped column names)
        rename_dict_1 = {}
        # Dictionary to reformat names into (collapsed) long form
        rename_dict_2 = {}
        # For casting column types
        astype_dict = {}
        # Columns to drop
        drops = []
        for col in self.included_cols():
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
        if self.col_included('t'):
            sort_order_1.append(bpd.to_list(self.reference_dict['t'])[0]) # Pre-reformatting
            sort_order_2.append(bpd.to_list(self.reference_dict['t'])[0][: - 1]) # Remove last number, e.g. t11 to t1

        # Append the last row if a mover (this is because the last observation is only given as an f2i, never as an f1i)
        last_obs_df = pd.DataFrame(self[self['m'] == 1]) \
            .sort_values(sort_order_1) \
            .drop_duplicates(subset='i', keep='last') \
            .rename(rename_dict_1, axis=1) # Sort by i, t to ensure last observation is actually last

        try:
            data_long = pd.concat([pd.DataFrame(self), last_obs_df], ignore_index=True) \
                .drop(drops, axis=1) \
                .rename(rename_dict_2, axis=1) \
                .astype(astype_dict)
        except ValueError: # If nan values, use Int8
            for col in astype_dict.keys():
                astype_dict[col] = 'Int64'
            data_long = pd.concat([pd.DataFrame(self), last_obs_df], ignore_index=True) \
                .drop(drops, axis=1) \
                .rename(rename_dict_2, axis=1) \
                .astype(astype_dict)
        # data_long = pd.DataFrame(self).groupby('i').apply(lambda a: a.append(a.iloc[-1].rename(rename_dict_1, axis=1)) if a.iloc[0]['m'] == 1 else a) \
        #     .reset_index(drop=True) \
        #     .drop(drops, axis=1) \
        #     .rename(rename_dict_2, axis=1) \
        #     .astype(astype_dict)

        # Sort columns and rows
        sorted_cols = sorted(data_long.columns, key=bpd.col_order)
        data_long = data_long[sorted_cols].sort_values(sort_order_2).reset_index(drop=True)

        if return_df:
            return data_long

        long_frame = self._constructor_long(data_long)
        long_frame.set_attributes(self, no_dict=True)

        return long_frame

    def unstack_es(self, return_df=False):
        '''
        Unstack (collapsed) event study data by stacking (and renaming) period 2 data below period 1 data for movers, then dropping period 2 columns, returning a (collapsed) long dataframe. Duplicates created from unstacking are dropped.

        Returns:
            long_frame (BipartiteLong(Collapsed) or Pandas DataFrame): BipartiteLong(Collapsed) or Pandas dataframe generated from (collapsed) event study data
        '''
        # Generate m column (the function checks if it already exists)
        self.gen_m()

        # Dictionary to swap names (necessary for last row of data, where period-2 observations are not located in subsequent period-1 column (as it doesn't exist), so must append the last row with swapped column names)
        rename_dict_1 = {}
        # Dictionary to reformat names into (collapsed) long form
        rename_dict_2 = {}
        # For casting column types
        astype_dict = {}
        # Columns to drop
        drops = []
        for col in self.included_cols():
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
        if self.col_included('t'):
            sort_order.append(bpd.to_list(self.reference_dict['t'])[0][: - 1]) # Remove last number, e.g. t11 to t1

        # Stack period 2 data if a mover (this is because the last observation is only given as an f2i, never as an f1i)
        stacked_df = pd.DataFrame(self[self['m'] == 1]).rename(rename_dict_1, axis=1)

        try:
            data_long = pd.concat([pd.DataFrame(self), stacked_df], ignore_index=True) \
                .drop(drops, axis=1) \
                .rename(rename_dict_2, axis=1) \
                .astype(astype_dict)
        except ValueError: # If nan values, use Int8
            for col in astype_dict.keys():
                astype_dict[col] = 'Int64'
            data_long = pd.concat([pd.DataFrame(self), stacked_df], ignore_index=True) \
                .drop(drops, axis=1) \
                .rename(rename_dict_2, axis=1) \
                .astype(astype_dict)

        # Sort columns and rows
        sorted_cols = sorted(data_long.columns, key=bpd.col_order)
        data_long = data_long[sorted_cols].sort_values(sort_order).reset_index(drop=True)

        if return_df:
            return data_long

        long_frame = self._constructor_long(data_long)
        long_frame.set_attributes(self, no_dict=True)

        return long_frame
