'''
Class for a bipartite network in long form
'''
import pandas as pd
import bipartitepandas as bpd

class BipartiteLong(bpd.BipartiteLongBase):
    '''
    Class for bipartite networks of firms and workers in long form. Inherits from BipartiteLongBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict or None): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        # Initialize DataFrame
        reference_dict = {'year': 'year'}
        super().__init__(*args, reference_dict=reference_dict, col_dict=col_dict, **kwargs)
        self.logger.info('BipartiteLong object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteLong

    def clean_data(self):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Returns:
            frame (BipartiteLong): BipartiteLong with cleaned data
        '''
        frame = bpd.BipartiteLongBase.clean_data(self)

        frame.logger.info('beginning BipartiteLong data cleaning')
        frame.logger.info('checking quality of data')
        frame.data_validity()

        frame.logger.info('BipartiteLong data cleaning complete')

        return frame

    def data_validity(self, inplace=True):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteLong): BipartiteLong with corrected columns and attributes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        success = True

        frame.logger.info('--- checking worker-year observations ---')
        max_obs = frame.groupby(['wid', 'year']).size().max()

        frame.logger.info('max number of worker-year observations (should be 1):' + str(max_obs))
        if max_obs > 1:
            success = False

        frame.logger.info('BipartiteLongBase success:' + str(success))

        return frame

    def get_collapsed_long(self):
        '''
        Collapse long data by job spells (so each spell for a particular worker at a particular firm is one observation).

        Returns:
            collapsed_frame (BipartiteLongCollapsed): BipartiteLongCollapsed object generated from long data collapsed by job spells
        '''
        # Copy data
        data = pd.DataFrame(self, copy=True)
        # Sort data by wid and year
        data = data.sort_values(['wid', 'year'])
        self.logger.info('copied data sorted by wid and year')
        # Determine whether m, cluster columns exist
        m = self.col_included('m')
        clustered = self.col_included('j')

        # Introduce lagged fid and wid
        data['fid_l1'] = data['fid'].shift(periods=1)
        data['wid_l1'] = data['wid'].shift(periods=1)
        self.logger.info('lagged fid introduced')

        # Generate spell ids
        # Source: https://stackoverflow.com/questions/59778744/pandas-grouping-and-aggregating-consecutive-rows-with-same-value-in-column
        new_spell = (data['fid'] != data['fid_l1']) | (data['wid'] != data['wid_l1']) # Allow for wid != wid_l1 to ensure that consecutive workers at the same firm get counted as different spells
        data['spell_id'] = new_spell.cumsum()
        self.logger.info('spell ids generated')

        # Aggregate at the spell level
        spell = data.groupby(['spell_id'])
        if m and clustered:
            data_spell = spell.agg(
                wid=pd.NamedAgg(column='wid', aggfunc='first'),
                comp=pd.NamedAgg(column='comp', aggfunc='mean'),
                fid=pd.NamedAgg(column='fid', aggfunc='first'),
                year_start=pd.NamedAgg(column='year', aggfunc='min'),
                year_end=pd.NamedAgg(column='year', aggfunc='max'),
                weight=pd.NamedAgg(column='wid', aggfunc='size'),
                m=pd.NamedAgg(column='m', aggfunc='first'),
                j=pd.NamedAgg(column='j', aggfunc='first')
            )
        elif m:
            data_spell = spell.agg(
                wid=pd.NamedAgg(column='wid', aggfunc='first'),
                comp=pd.NamedAgg(column='comp', aggfunc='mean'),
                fid=pd.NamedAgg(column='fid', aggfunc='first'),
                year_start=pd.NamedAgg(column='year', aggfunc='min'),
                year_end=pd.NamedAgg(column='year', aggfunc='max'),
                weight=pd.NamedAgg(column='wid', aggfunc='size'),
                m=pd.NamedAgg(column='m', aggfunc='first')
            )
        elif clustered:
            data_spell = spell.agg(
                wid=pd.NamedAgg(column='wid', aggfunc='first'),
                comp=pd.NamedAgg(column='comp', aggfunc='mean'),
                fid=pd.NamedAgg(column='fid', aggfunc='first'),
                year_start=pd.NamedAgg(column='year', aggfunc='min'),
                year_end=pd.NamedAgg(column='year', aggfunc='max'),
                weight=pd.NamedAgg(column='wid', aggfunc='size'),
                j=pd.NamedAgg(column='j', aggfunc='first')
            )
        else:
            data_spell = spell.agg(
                wid=pd.NamedAgg(column='wid', aggfunc='first'),
                comp=pd.NamedAgg(column='comp', aggfunc='mean'),
                fid=pd.NamedAgg(column='fid', aggfunc='first'),
                year_start=pd.NamedAgg(column='year', aggfunc='min'),
                year_end=pd.NamedAgg(column='year', aggfunc='max'),
                weight=pd.NamedAgg(column='wid', aggfunc='size')
            )
        # Classify movers and stayers
        if not m:
            spell_count = data_spell.groupby(['wid']).transform('count')['fid'] # Choice of fid arbitrary
            data_spell['m'] = (spell_count > 1).astype(int)
        collapsed_data = data_spell.reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(collapsed_data.columns, key=bpd.col_order)
        collapsed_data = collapsed_data[sorted_cols]

        self.logger.info('data aggregated at the spell level')

        collapsed_frame = bpd.BipartiteLongCollapsed(collapsed_data)
        collapsed_frame.set_attributes(self, no_dict=True)

        return collapsed_frame

    def get_es(self):
        '''
        Return long form data reformatted into event study data.

        Returns:
            es_frame (BipartiteEventStudy): BipartiteEventStudy object generated from long data
        '''
        data_es = self.copy()

        # Determine whether m, cluster columns exist
        m = data_es.col_included('m')
        clustered = data_es.col_included('j')

        if not m:
            # Generate m column
            data_es.gen_m()

        # Convert into a dataframe
        data_es = pd.DataFrame(data_es)

        # Add lagged values
        data_es = data_es.sort_values(['wid', 'year'])
        data_es['fid_l1'] = data_es['fid'].shift(periods=1)
        data_es['wid_l1'] = data_es['wid'].shift(periods=1) # Used to mark consecutive observations as being for the same worker
        data_es['comp_l1'] = data_es['comp'].shift(periods=1)
        data_es['year_l1'] = data_es['year'].shift(periods=1)
        if clustered:
            data_es['j_l1'] = data_es['j'].shift(periods=1)
        data_es = data_es[data_es['wid'] == data_es['wid_l1']]
        data_es[['fid_l1', 'year_l1']] = data_es[['fid_l1', 'year_l1']].astype(int) # Shifting adds nans which converts columns into float, but want int
        
        data_es = data_es.rename({
            'fid_l1': 'f1i',
            'fid': 'f2i',
            'comp_l1': 'y1',
            'comp': 'y2',
            'year': 'year_2',
            'year_l1': 'year_1',
        }, axis=1)

        keep_cols = ['wid', 'y1', 'y2', 'f1i', 'f2i', 'year_1', 'year_2', 'm']

        if clustered:
            data_es['j_l1'] = data_es['j_l1'].astype(int)
            data_es = data_es.rename({'j': 'j2', 'j_l1': 'j1'}, axis=1)
            keep_cols += ['j1', 'j2']

        # Keep only relevant columns
        data_es = data_es[keep_cols]
        self.logger.info('columns updated')

        # Sort columns
        sorted_cols = sorted(data_es.columns, key=bpd.col_order)
        data_es = data_es[sorted_cols].reset_index(drop=True)

        self.logger.info('data reformatted as event study')

        es_frame = bpd.BipartiteEventStudy(data_es)
        es_frame.set_attributes(self, no_dict=True)

        return es_frame
