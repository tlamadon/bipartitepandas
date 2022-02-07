'''
Bipartite DataFrame general constructor.
'''
import pandas as pd
import bipartitepandas as bpd

class BipartiteDataFrame():
    '''
    Constructor class for interacting with BipartitePandas dataframe classes more easily.

    Arguments:
        *args: arguments for Pandas DataFrame
        ##### ANY FORMAT COLUMNS #####
        i (NumPy Array or Pandas Series of any type): worker id (required)
        ##### BASE LONG FORMAT COLUMNS #####
        j (NumPy Array or Pandas Series of any type): firm id (required)
        y (NumPy Array or Pandas Series of floats or ints): income (required)
        ##### NON-COLLAPSED LONG FORMAT COLUMNS #####
        t (NumPy Array or Pandas Series of ints): time (optional)
        g (NumPy Array or Pandas Series of any type): firm cluster (optional)
        w (NumPy Array or Pandas Series of floats or ints): observation weight (optional)
        ##### COLLAPSED LONG FORMAT COLUMNS #####
        t1 (NumPy Array or Pandas Series of ints): first time in worker-firm spell (optional)
        t2 (NumPy Array or Pandas Series of ints): last time in worker-firm spell (optional)
        ##### BASE EVENT STUDY FORMAT COLUMNS #####
        j1 (NumPy Array or Pandas Series of any type): firm id in first period of event study (required)
        j2 (NumPy Array or Pandas Series of any type): firm id in second period of event study (required)
        y1 (NumPy Array or Pandas Series of floats or ints): income in first period of event study (required)
        y2 (NumPy Array or Pandas Series of floats or ints): income in second period of event study (required)
        g1 (NumPy Array or Pandas Series of any type): firm cluster in first period of event study (optional)
        g2 (NumPy Array or Pandas Series of any type): firm cluster in second period of event study (optional)
        w1 (NumPy Array or Pandas Series of floats or ints): observation weight in first period of event study (optional)
        w2 (NumPy Array or Pandas Series of floats or ints): observation weight in second period of event study (optional)
        ##### NON-COLLAPSED EVENT STUDY FORMAT COLUMNS #####
        t1 (NumPy Array or Pandas Series of ints): time in first period of event study (optional)
        t2 (NumPy Array or Pandas Series of ints): time in second period of event study (optional)
        ##### COLLAPSED EVENT STUDY FORMAT COLUMNS #####
        t11 (NumPy Array or Pandas Series of ints): first time in worker-firm spell in first period of event study (optional)
        t12 (NumPy Array or Pandas Series of ints): last time in worker-firm spell in first period of event study (optional)
        t21 (NumPy Array or Pandas Series of ints): first time in worker-firm spell in second period of event study (optional)
        t22 (NumPy Array or Pandas Series of ints): last time in worker-firm spell in second period of event study (optional)
        **kwargs: keyword arguments for BipartiteBase
    '''

    def __init__(self):
        '''
        Do nothing. All work is done in __new__(). Source: https://stackoverflow.com/a/2491881/17333120.
        '''
        pass

    def __new__(self, *args, i, j=None, j1=None, j2=None, y=None, y1=None, y2=None, t=None, t1=None, t2=None, t11=None, t12=None, t21=None, t22=None, g=None, g1=None, g2=None, w=None, w1=None, w2=None, **kwargs):
        if j is not None:
            ### Base long format ###
            if y is None:
                # y column is required
                raise NotImplementedError("At least one required column is missing. A BipartiteDataFrame requires either a 'y' column or 'y1' and 'y2' columns, indicating income for each observation. Please input only 'y' for long format, or 'y1' and 'y2' for event study format.")
            elif (y1 is not None) or (y2 is not None):
                # Can't mix base long and base event study for y
                raise NotImplementedError("A BipartiteDataFrame requires either a 'y' column or 'y1' and 'y2' columns, indicating income for each observation, but cannot include both. Please input only 'y' for long format, or 'y1' and 'y2' for event study format.")
            if (j1 is not None) or (j2 is not None):
                # Can't mix base long and base event study for j
                raise NotImplementedError("A BipartiteDataFrame requires either a 'j' column or 'j1' and 'j2' columns, indicating firm ids for each observation, but cannot include both. Please input only 'j' for long format, or 'j1' and 'j2' for event study format.")
            if (g1 is not None) or (g2 is not None):
                # Can't mix base long and base event study for g
                raise NotImplementedError("A BipartiteDataFrame can optionally include either a 'g' column or 'g1' and 'g2' columns, indicating firm clusters for each observation, but cannot include both. Please input only 'g' for long format, or 'g1' and 'g2' for event study format.")
            if (w1 is not None) or (w2 is not None):
                # Can't mix base long and base event study for w
                raise NotImplementedError("A BipartiteDataFrame can optionally include either a 'w' column or 'w1' and 'w2' columns, indicating a weight for each observation, but cannot include both. Please input only 'w' for long format, or 'w1' and 'w2' for event study format.")
            if t is not None:
                ## Long format ##
                if (t1 is not None) or (t2 is not None):
                    # Can't mix long and collapsed long for t
                    raise NotImplementedError("A BipartiteDataFrame with 'j' and 'y' columns is long or collapsed long format. Long format requires a 't' column, while collapsed long format requires 't1' and 't2' columns. However, the constructor will not work if both are included.")
                elif (t11 is not None) or (t12 is not None) or (t21 is not None) or (t22 is not None):
                    # Can't mix long and collapsed event study for t
                    raise NotImplementedError("A BipartiteDataFrame with 'j', 'y', and 't' columns is long format. The constructor will not work if 't11', 't12', 't21', or 't22' columns are included.")
                ##### RETURN LONG #####
                df = pd.DataFrame({'i': i, 'j': j, 'y': y, 't': t})
                if g is not None:
                    df.loc[:, 'g'] = g
                if w is not None:
                    df.loc[:, 'w'] = w
                return bpd.BipartiteLong(df, **kwargs)
            elif (t1 is not None) and (t2 is not None):
                ## Collapsed long format ##
                if (t11 is not None) or (t12 is not None) or (t21 is not None) or (t22 is not None):
                    # Can't mix collapsed long and collapsed event study for t
                    raise NotImplementedError("A BipartiteDataFrame with 'j', 'y', 't1', and 't2' columns is collapsed long format. The constructor will not work if 't11', 't12', 't21', or 't22' columns are included.")
                ##### RETURN COLLAPSED LONG #####
                df = pd.DataFrame({'i': i, 'j': j, 'y': y, 't1': t1, 't2': t2})
                if g is not None:
                    df.loc[:, 'g'] = g
                if w is not None:
                    df.loc[:, 'w'] = w
                return bpd.BipartiteLongCollapsed(df, **kwargs)
            elif (t1 is not None) or (t2 is not None):
                # Can't include only one of t1 and t2 for collapsed long
                raise NotImplementedError("A BipartiteDataFrame with 'j', 'y', 't1', and 't2' columns is collapsed long format. The constructor will not work if only one of 't1' and 't2' is included.")
            elif (t11 is not None) or (t12 is not None) or (t21 is not None) or (t22 is not None):
                # Can't mix base long and collapsed event study for t
                raise NotImplementedError("A BipartiteDataFrame with 'j' and 'y' columns is long or collapsed long format. The constructor will not work if 't11', 't12', 't21', or 't22' columns are included.")
            ##### RETURN UNSPECIFIED LONG #####
            df = pd.DataFrame({'i': i, 'j': j, 'y': y})
            if g is not None:
                df.loc[:, 'g'] = g
            if w is not None:
                df.loc[:, 'w'] = w
            return bpd.BipartiteLong(df, **kwargs)
        elif (j1 is not None) and (j2 is not None):
            ### Base event study format ###
            if (y1 is None) or (y2 is None):
                # y1 and y2 columns are required
                raise NotImplementedError("At least one required column is missing. A BipartiteDataFrame requires either a 'y' column or 'y1' and 'y2' columns, indicating income for each observation. Please input only 'y' for long format, or 'y1' and 'y2' for event study format.")
            elif y is not None:
                # Can't mix base long and base event study for y
                raise NotImplementedError("A BipartiteDataFrame requires either a 'y' column or 'y1' and 'y2' columns, indicating income for each observation, but cannot include both. Please input only 'y' for long format, or 'y1' and 'y2' for event study format.")
            if g is not None:
                # Can't mix base long and base event study for g
                raise NotImplementedError("A BipartiteDataFrame can optionally include either a 'g' column or 'g1' and 'g2' columns, indicating firm clusters for each observation, but cannot include both. Please input only 'g' for long format, or 'g1' and 'g2' for event study format.")
            if (g1 is not None) != (g2 is not None):
                # Can't include only one of g1 and g2 for event study
                raise NotImplementedError("A BipartiteDataFrame with 'j1', 'j2', 'y1', and 'y2' columns is either event study or collapsed event study format. The constructor will not work if only one of 'g1' and 'g2' is included.")
            if w is not None:
                # Can't mix base long and base event study for w
                raise NotImplementedError("A BipartiteDataFrame can optionally include either a 'w' column or 'w1' and 'w2' columns, indicating a weight for each observation, but cannot include both. Please input only 'w' for long format, or 'w1' and 'w2' for event study format.")
            if (w1 is not None) != (w2 is not None):
                # Can't include only one of w1 and w2 for event study
                raise NotImplementedError("A BipartiteDataFrame with 'j1', 'j2', 'y1', and 'y2' columns is either event study or collapsed event study format. The constructor will not work if only one of 'w1' and 'w2' is included.")
            elif (t1 is not None) and (t2 is not None):
                ## Event study format ##
                if (t11 is not None) or (t12 is not None) or (t21 is not None) or (t22 is not None):
                    # Can't mix event study and collapsed event study for t
                    raise NotImplementedError("A BipartiteDataFrame with 'j1', 'j2', 't1', and 't2' columns is event study format. The constructor will not work if 't11', 't12', 't21', or 't22' columns are included.")
                ##### RETURN EVENT STUDY #####
                df = pd.DataFrame({'i': i, 'j1': j1, 'j2': j2, 'y1': y1, 'y2': y2, 't1': t1, 't2': t2})
                if (g1 is not None) and (g2 is not None):
                    df.loc[:, 'g1'] = g1
                    df.loc[:, 'g2'] = g2
                if (w1 is not None) and (w2 is not None):
                    df.loc[:, 'w1'] = w1
                    df.loc[:, 'w2'] = w2
                return bpd.BipartiteEventStudy(df, **kwargs)
            elif (t1 is not None) or (t2 is not None):
                # Can't include only one of t1 and t2 for event study
                raise NotImplementedError("A BipartiteDataFrame with 'j1', 'j2', 't1', and 't2' columns is event study format. The constructor will not work if only one of 't1' and 't2' is included.")
            elif (t11 is not None) and (t12 is not None) and (t21 is not None) and (t22 is not None):
                ## Collapsed event study format ##
                if (t1 is not None) or (t2 is not None):
                    # Can't mix event study and collapsed event study for t
                    raise NotImplementedError("A BipartiteDataFrame with 'j1', 'j2', 't11', 't12', 't21', and 't22' columns is collapsed event study format. The constructor will not work if 't1' or 't2' columns are included.")
                ##### RETURN COLLAPSED EVENT STUDY #####
                df = pd.DataFrame({'i': i, 'j1': j1, 'j2': j2, 'y1': y1, 'y2': y2, 't11': t11, 't12': t12, 't21': t21, 't22': t22})
                if (g1 is not None) and (g2 is not None):
                    df.loc[:, 'g1'] = g1
                    df.loc[:, 'g2'] = g2
                if (w1 is not None) and (w2 is not None):
                    df.loc[:, 'w1'] = w1
                    df.loc[:, 'w2'] = w2
                return bpd.BipartiteEventStudyCollapsed(df, **kwargs)
            elif (t11 is not None) or (t12 is not None) or (t21 is not None) or (t22 is not None):
                # Can't include only one of t11, t12, t21, and t22 for collapsed event study
                raise NotImplementedError("A BipartiteDataFrame with 'j1', 'j2', 't11', 't12', 't21', and 't22' columns is collapsed event study format. The constructor will not work if only one of 't11', 't12', 't21', and 't22' is included.")
            ##### RETURN UNSPECIFIED EVENT STUDY #####
            df = pd.DataFrame({'i': i, 'j1': j1, 'j2': j2, 'y1': y1, 'y2': y2})
            if (g1 is not None) and (g2 is not None):
                df.loc[:, 'g1'] = g1
                df.loc[:, 'g2'] = g2
            if (w1 is not None) and (w2 is not None):
                df.loc[:, 'w1'] = w1
                df.loc[:, 'w2'] = w2
            return bpd.BipartiteEventStudy(df, **kwargs)
        else:
            # Neither long format nor event study format
            raise NotImplementedError("A BipartiteDataFrame requires either a 'j' column or 'j1' and 'j2' columns, indicating firm ids for each observation. Please input only 'j' for long format, or 'j1' and 'j2' for event study format.")
