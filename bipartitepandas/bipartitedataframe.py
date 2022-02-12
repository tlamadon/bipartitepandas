'''
Bipartite DataFrame general constructor.
'''
from multiprocessing.sharedctypes import Value
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
        m (NumPy Array or Pandas Series of ints): mover value (optional)
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
        ##### CUSTOM COLUMNS #####
        custom_contig_dict (dict of bool): for new columns, optionally link general column names to whether that set of columns is contiguous (e.g. 'j' is firm ids, links to columns 'j1' and 'j2' and should be contiguous; then set {'j': True})
        custom_dtype_dict (dict of str): for new columns, optionally link general column names to the datatype for that set of columns (e.g. 'y' is income, links to columns 'y1' and 'y2' and should be float; then set {'y': 'float'}); must be one of 'int', 'float', or 'any'
        custom_how_collapse_dict (dict of (function or str or None)): for new columns, optionally link general column names to how members of that set of columns should be collapsed at the worker-firm spell level (e.g. 'y' is income, links to columns 'y1' and 'y2' and should become the mean at the worker-firm spell level; then set {'y': 'float'}); must be a valid input for Pandas groupby; if None, column will be dropped during collapse/uncollapse
        custom_long_es_split_dict (dict of (bool or None)): for new columns, optionally link general column names to whether members of that set of columns should split into two when converting from long to event study; if None, columns will be dropped when converting between (collapsed) long and (collapsed) event study formats
        **kwargs: keyword arguments for BipartiteBase, or new columns
    '''

    def __init__(self):
        '''
        Do nothing. All work is done in __new__(). Source: https://stackoverflow.com/a/2491881/17333120.
        '''
        pass

    def __new__(self, *args, i, j=None, j1=None, j2=None, y=None, y1=None, y2=None, t=None, t1=None, t2=None, t11=None, t12=None, t21=None, t22=None, g=None, g1=None, g2=None, w=None, w1=None, w2=None, m=None, custom_contig_dict={}, custom_dtype_dict={}, custom_how_collapse_dict={}, custom_long_es_split_dict={}, **kwargs):
        # Dataframe to fill in
        df = None
        # Figure out which kwargs are new columns
        new_kwargs = {}
        new_cols = {}
        new_cols_reference_dict = {}
        for k, v in kwargs.items():
            if bpd.util._is_subtype(v, bpd.util.col_type):
                col_name, col_num = bpd.util._text_num_split(k)
                if col_name in new_cols.keys():
                    # If general column already seen
                    new_cols[col_name].append(v)
                    new_cols_reference_dict[col_name].append(k)
                else:
                    # If new general column
                    new_cols[col_name] = [v]
                    new_cols_reference_dict[col_name] = [k]
                    ## Figure out dictionary values
                    # If is_contig is not specified, default to False
                    if col_name not in custom_contig_dict.keys():
                        custom_contig_dict[col_name] = False
                    # If long_es_split is not specified, default to True
                    if col_name not in custom_long_es_split_dict.keys():
                        custom_long_es_split_dict[col_name] = True
                    # Since new general column, try to infer column dtype and how_collapse
                    col_float_int = int(bpd.util._is_subdtype(v, 'float')) + int(bpd.util._is_subdtype(v, 'int'))
                    dtype_how_collapse_dict = {
                        0: { # If neither float nor int
                            'dtype': 'any',
                            'how_collapse': 'first'
                        },
                        1: { # If float but not int
                            'dtype': 'float',
                            'how_collapse': 'mean'
                        },
                        2: { # If int and float
                            'dtype': 'int',
                            'how_collapse': 'first'
                        }
                    }
                    if col_name not in custom_dtype_dict.keys():
                        # If dtype isn't manually set, infer what it should be
                        custom_dtype_dict[col_name] = dtype_how_collapse_dict[col_float_int]['dtype']
                    if col_name not in custom_how_collapse_dict.keys():
                        # If how_collapse isn't manually set, infer what it should be
                        custom_how_collapse_dict[col_name] = dtype_how_collapse_dict[col_float_int]['how_collapse']
            else:
                new_kwargs[k] = v
        if j is not None:
            ### Base long format ###
            if (j1 is not None) or (j2 is not None):
                # Can't mix base long and base event study for j
                raise ValueError("A long format dataframe requires a 'j' column, indicating a firm id for each observation, while an event study format dataframe requires 'j1' and 'j2' columns, indicating firm ids for the the pre- and post- periods for each observation in the event study. However, your input includes 'j' in addition to at least one of 'j1' and 'j2'. Please include only the set of firm id columns relevant for the format you would like to use.")
            if y is None:
                # y column is required
                raise ValueError("A long format dataframe requires a 'j' column, indicating a firm id for each observation, while an event study format dataframe requires 'j1' and 'j2' columns, indicating firm ids for the pre- and post- periods for each observation in the event study. Your input includes a 'j' column, indicating the construction of a long format dataframe. However, a long format dataframe also requires a 'y' column, indicating income for each observation. Your input does not include a 'y' column, so please make sure to include this column, or swap the 'j' column for 'j1' and 'j2' columns if you would like to construct an event study format dataframe.")
            elif (y1 is not None) or (y2 is not None):
                # Can't mix base long and base event study for y
                raise ValueError("Your input includes a 'j' column, indicating the construction of a long format dataframe. A long format dataframe requires a 'y' column, indicating income for each observation, while an event study format dataframe requires 'y1' and 'y2' columns, indicating income for the the pre- and post- periods for each observation in the event study. However, your input includes 'y' in addition to at least one of 'y1' and 'y2'. Please input only 'y' for long format.")
            if (t11 is not None) or (t12 is not None) or (t21 is not None) or (t22 is not None):
                # Can't mix (collapsed) long and collapsed event study for t
                raise ValueError("Your input includes 'j' and 'y' columns, indicating the construction of a long or collapsed long dataframe. A long dataframe can optionally include a 't' column, indicating the time period for each observation, while a collapsed long dataframe can optionally include 't1' and 't2' columns, indicating the first and last time period, respectively, for the worker-firm spell represented by each observation. Similarly, an event study dataframe can optionally include 't1' and 't2' columns, indicating the time for the the pre- and post- periods for each observation in the event study, while a collapsed event study dataframe can optionally include 't11', 't12', 't21', and 't22' columns, indicating the first and last time period, respectively, for the worker-firm spells represented by the pre- and post- periods for each observation in the event study. However, your input includes at least one of 't11', 't12', 't21' and 't22'. Please optionally input only 't' for long format, or 't1' and 't2' for collapsed long format.")
            if (g1 is not None) or (g2 is not None):
                # Can't mix base long and base event study for g
                raise ValueError("Your input includes 'j' and 'y' columns, indicating the construction of a long format dataframe. A long format dataframe can optionally include a 'g' column, indicating firm clusters for each observation, while an event study format dataframe can optionally include 'g1' and 'g2' columns, indicating firm clusters for the the pre- and post- periods for each observation in the event study. However, your input includes at least one of 'g1' and 'g2'. Please remove these inputs for long format.")
            if (w1 is not None) or (w2 is not None):
                # Can't mix base long and base event study for w
                raise ValueError("Your input includes 'j' and 'y' columns, indicating the construction of a long format dataframe. A long format dataframe can optionally include a 'w' column, indicating a weight for each observation, while an event study format dataframe can optionally include 'w1' and 'w2' columns, indicating weights for the the pre- and post- periods for each observation in the event study. However, your input includes at least one of 'w1' and 'w2'. Please remove these inputs for long format.")
            if t is not None:
                ## Long format ##
                if (t1 is not None) or (t2 is not None):
                    # Can't mix long and collapsed long for t
                    raise ValueError("Your input includes 'j' and 'y' columns, indicating the construction of a long or collapsed long dataframe. A long dataframe can optionally include a 't' column, indicating the time period for each observation, while a collapsed long dataframe can optionally include 't1' and 't2' columns, indicating the first and last time period, respectively, for the worker-firm spell represented by each observation. However, your input includes 't' in addition to at least one of 't1' and 't2'. Please include only the set of time columns relevant for the format you would like to use.")
                ##### RETURN LONG #####
                df = pd.DataFrame({'i': i, 'j': j, 'y': y, 't': t})
                if g is not None:
                    df.loc[:, 'g'] = g
                if w is not None:
                    df.loc[:, 'w'] = w
                if m is not None:
                    df.loc[:, 'm'] = m
                df = bpd.BipartiteLong(*args, df, **new_kwargs)
            elif (t1 is not None) and (t2 is not None):
                ## Collapsed long format ##
                ##### RETURN COLLAPSED LONG #####
                df = pd.DataFrame({'i': i, 'j': j, 'y': y, 't1': t1, 't2': t2})
                if g is not None:
                    df.loc[:, 'g'] = g
                if w is not None:
                    df.loc[:, 'w'] = w
                if m is not None:
                    df.loc[:, 'm'] = m
                df = bpd.BipartiteLongCollapsed(*args, df, **new_kwargs)
            elif (t1 is not None) or (t2 is not None):
                # Can't include only one of t1 and t2 for collapsed long
                raise ValueError("Your input includes 'j' and 'y' columns, indicating the construction of a long or collapsed long dataframe. A long dataframe can optionally include a 't' column, indicating the time period for each observation, while a collapsed long dataframe can optionally include 't1' and 't2' columns, indicating the first and last time period, respectively, for the worker-firm spell represented by each observation. However, your input includes only one of 't1' and 't2'. Please rename your column to 't' for long format, add the missing time column for collapsed long format, or remove the included time column, as time columns are optional.")
            if df is None:
                ##### RETURN UNSPECIFIED LONG #####
                df = pd.DataFrame({'i': i, 'j': j, 'y': y})
                if g is not None:
                    df.loc[:, 'g'] = g
                if w is not None:
                    df.loc[:, 'w'] = w
                if m is not None:
                    df.loc[:, 'm'] = m
                df = bpd.BipartiteLong(*args, df, **new_kwargs)
        elif (j1 is not None) and (j2 is not None):
            ### Base event study format ###
            if (y1 is None) or (y2 is None):
                # y1 and y2 columns are required
                raise ValueError("A long format dataframe requires a 'j' column, indicating a firm id for each observation, while an event study format dataframe requires 'j1' and 'j2' columns, indicating firm ids for the pre- and post- periods for each observation in the event study. Your input includes 'j1' and 'j2' columns, indicating the construction of an event study format dataframe. However, an event study format dataframe also requires 'y1' and 'y2' columns, indicating income for the pre- and post- periods for each observation in the event study. Your input includes at most one of 'y1' and 'y2', so please make sure to include both of these columns, or swap the 'j1' and 'j2' columns for a 'j' column if you would like to construct a long format dataframe.")
            elif y is not None:
                # Can't mix base long and base event study for y
                raise ValueError("Your input includes 'j1' and 'j2' columns, indicating the construction of an event study format dataframe. A long format dataframe requires a 'y' column, indicating income for each observation, while an event study format dataframe requires 'y1' and 'y2' columns, indicating income for the the pre- and post- periods for each observation in the event study. However, your input includes 'y1' and 'y2' in addition to 'y'. Please input only 'y1' and 'y2' for event study format.")
            if g is not None:
                # Can't mix base long and base event study for g
                raise ValueError("Your input includes 'j1', 'j2', 'y1', and 'y2' columns, indicating the construction of an event study format dataframe. A long format dataframe can optionally include a 'g' column, indicating firm clusters for each observation, while an event study format dataframe can optionally include 'g1' and 'g2' columns, indicating firm clusters for the the pre- and post- periods for each observation in the event study. However, your input includes 'g'. Please remove this input for event study format.")
            if (g1 is not None) != (g2 is not None):
                # Can't include only one of g1 and g2 for event study
                raise ValueError("Your input includes 'j1', 'j2', 'y1', and 'y2' columns, indicating the construction of an event study format dataframe. A long format dataframe can optionally include a 'g' column, indicating firm clusters for each observation, while an event study format dataframe can optionally include 'g1' and 'g2' columns, indicating firm clusters for the the pre- and post- periods for each observation in the event study. However, your input includes only one of 'g1' and 'g2'. Please make sure to include either none or both of these columns.")
            if w is not None:
                # Can't mix base long and base event study for w
                raise ValueError("Your input includes 'j1', 'j2', 'y1', and 'y2' columns, indicating the construction of an event study format dataframe. A long format dataframe can optionally include a 'w' column, indicating a weight for each observation, while an event study format dataframe can optionally include 'w1' and 'w2' columns, indicating weights for the the pre- and post- periods for each observation in the event study. However, your input includes 'w'. Please remove this input for event study format.")
            if (w1 is not None) != (w2 is not None):
                # Can't include only one of w1 and w2 for event study
                raise ValueError("Your input includes 'j1', 'j2', 'y1', and 'y2' columns, indicating the construction of an event study format dataframe. A long format dataframe can optionally include a 'w' column, indicating a weight for each observation, while an event study format dataframe can optionally include 'w1' and 'w2' columns, indicating weights for the the pre- and post- periods for each observation in the event study. However, your input includes only one of 'w1' and 'w2'. Please make sure to include either none or both of these columns.")
            elif (t1 is not None) and (t2 is not None):
                ## Event study format ##
                if (t11 is not None) or (t12 is not None) or (t21 is not None) or (t22 is not None):
                    # Can't mix event study and collapsed event study for t
                    raise ValueError("Your input includes 'j1', 'j2', 'y1', and 'y2' columns, indicating the construction of an event study or collapsed event study dataframe. An event study dataframe can optionally include 't1' and 't2' columns, indicating the time for the the pre- and post- periods for each observation in the event study, while a collapsed event study dataframe can optionally include 't11', 't12', 't21', and 't22' columns, indicating the first and last time period, respectively, for the worker-firm spells represented by the pre- and post- periods for each observation in the event study. However, your input includes 't1' and 't2' in addition to at least one of 't11', 't12', 't21', and 't22'. Please include only the set of time columns relevant for the format you would like to use.")
                ##### RETURN EVENT STUDY #####
                df = pd.DataFrame({'i': i, 'j1': j1, 'j2': j2, 'y1': y1, 'y2': y2, 't1': t1, 't2': t2})
                if (g1 is not None) and (g2 is not None):
                    df.loc[:, 'g1'] = g1
                    df.loc[:, 'g2'] = g2
                if (w1 is not None) and (w2 is not None):
                    df.loc[:, 'w1'] = w1
                    df.loc[:, 'w2'] = w2
                if m is not None:
                    df.loc[:, 'm'] = m
                df = bpd.BipartiteEventStudy(*args, df, **new_kwargs)
            elif (t1 is not None) or (t2 is not None):
                # Can't include only one of t1 and t2 for event study
                raise ValueError("Your input includes 'j1', 'j2', 'y1', and 'y2' columns, indicating the construction of an event study or collapsed event study dataframe. An event study dataframe can optionally include 't1' and 't2' columns, indicating the time for the the pre- and post- periods for each observation in the event study, while a collapsed event study dataframe can optionally include 't11', 't12', 't21', and 't22' columns, indicating the first and last time period, respectively, for the worker-firm spells represented by the pre- and post- periods for each observation in the event study. However, your input includes only one of 't1' and 't2'. Please add the missing time column for event study format or remove the included time column, as time columns are optional.")
            elif (t11 is not None) and (t12 is not None) and (t21 is not None) and (t22 is not None):
                ## Collapsed event study format ##
                ##### RETURN COLLAPSED EVENT STUDY #####
                df = pd.DataFrame({'i': i, 'j1': j1, 'j2': j2, 'y1': y1, 'y2': y2, 't11': t11, 't12': t12, 't21': t21, 't22': t22})
                if (g1 is not None) and (g2 is not None):
                    df.loc[:, 'g1'] = g1
                    df.loc[:, 'g2'] = g2
                if (w1 is not None) and (w2 is not None):
                    df.loc[:, 'w1'] = w1
                    df.loc[:, 'w2'] = w2
                if m is not None:
                    df.loc[:, 'm'] = m
                df = bpd.BipartiteEventStudyCollapsed(*args, df, **new_kwargs)
            elif (t11 is not None) or (t12 is not None) or (t21 is not None) or (t22 is not None):
                # Can't include only one of t11, t12, t21, and t22 for collapsed event study
                raise ValueError("Your input includes 'j1', 'j2', 'y1', and 'y2' columns, indicating the construction of an event study or collapsed event study dataframe. An event study dataframe can optionally include 't1' and 't2' columns, indicating the time for the the pre- and post- periods for each observation in the event study, while a collapsed event study dataframe can optionally include 't11', 't12', 't21', and 't22' columns, indicating the first and last time period, respectively, for the worker-firm spells represented by the pre- and post- periods for each observation in the event study. However, your input includes a strict (and nonempty) subset of 't11', 't12', 't21', and 't22'. Please add the missing time column(s) for collapsed event study format or remove the included time column(s), as time columns are optional.")
            if df is None:
                ##### RETURN UNSPECIFIED EVENT STUDY #####
                df = pd.DataFrame({'i': i, 'j1': j1, 'j2': j2, 'y1': y1, 'y2': y2})
                if (g1 is not None) and (g2 is not None):
                    df.loc[:, 'g1'] = g1
                    df.loc[:, 'g2'] = g2
                if (w1 is not None) and (w2 is not None):
                    df.loc[:, 'w1'] = w1
                    df.loc[:, 'w2'] = w2
                if m is not None:
                    df.loc[:, 'm'] = m
                df = bpd.BipartiteEventStudy(*args, df, **new_kwargs)
        else:
            # Neither long format nor event study format
            raise ValueError("A long format dataframe requires a 'j' column, indicating a firm id for each observation, while an event study format dataframe requires 'j1' and 'j2' columns, indicating firm ids for the the pre- and post- periods for each observation in the event study. However, your input does not include 'j' and includes at most one of 'j1' and 'j2'. Please make sure to include the set of firm id columns relevant for the format you would like to use.")

        if len(new_cols) > 0:
            # If new columns
            for new_col_name, new_col_data in new_cols.items():
                col_reference = bpd.util.to_list(new_cols_reference_dict[new_col_name])
                if len(col_reference) == 1:
                    # Constructed col_references are forced to be lists, if it's length one then just extract the single value from the list
                    col_reference = col_reference[0]
                df = df.add_column(new_col_data, new_col_name, col_reference=col_reference, is_contig=custom_contig_dict[new_col_name], dtype=custom_dtype_dict[new_col_name], how_collapse=custom_how_collapse_dict[new_col_name], long_es_split=custom_long_es_split_dict[new_col_name], copy=False)

        return df
