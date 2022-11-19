import pandas as pd
import re
import pandas_profiling as pp

#region main.py
# from ez_pandas import *
# file_name = 'data/Pokemon.csv'
# df = read_csv(file_name, ',')
#endregion

#region Importing/Exporting
def read_csv(file_name: str, separator: str):
    return pd.read_csv(file_name, sep=separator)

def saving_csv(df: pd.DataFrame, file_name: str):
    df.to_csv(f"{file_name}.csv", index=False)

def saving_xlsx(df: pd.DataFrame, file_name: str):
    df.to_excel(f"{file_name}.xlsx", index=False)

def from_clipboard(separator: str):
    return pd.read_clipboard(sep=separator)

def reading_by_chunk_and_grouping(df: pd.DataFrame, group_by_column: str):
    new_df = pd.DataFrame(columns=df.columns)
    # chunksize = rows
    for df in pd.read_csv(file_name, chunksize=5):
        result = df.groupby([group_by_column]).count()
        new_df = pd.concat([new_df, result])
#endregion

#region Printing
def print_df(df: pd.DataFrame):
    print(df.to_string())

def print_beginning(df: pd.DataFrame, rows: int):
    print(df.head(rows).to_string())

def print_ending(df: pd.DataFrame, rows: int):
    print(df.tail(rows).to_string())
#endregion

#region DF Info
def get_headers(df: pd.DataFrame):
    return df.columns

def describe(df: pd.DataFrame):
    return df.describe().to_string()

def shape(df: pd.DataFrame):
    return df.shape

def dtypes(df: pd.DataFrame):
    return df.dtypes

def versions():
    return pd._version, pd.show_versions()

def missing_data(df: pd.DataFrame):
    return df.isnull().sum(), df.isna().sum()

def profile_df(df: pd.DataFrame):
    pp.ProfileReport(df)
#endregion

#region Reference
def worth_mentioning():
    text = """"
    reverse groupby: .unstack()
    continous to category: pd.cut

    apply example:
    def func(row):
        return (f"{row['col1']} - row['col2']")
    df.apply(func, axis = 1)
    df['new_col'] = df['col'].apply(func, args=[10, 100])
"""
    return text
#endregion

#region Get
def get_value(df: pd.DataFrame, row: int, column: int):
    # print(df.iloc[2, 2])
    return df.iloc[row, column]

def get_column(df: pd.DataFrame, columns: list):
    # df[['Attack', 'Defense']]
    return df[columns]

def get_row_by_index(df: pd.DataFrame, index: int):
    # print(df.iloc[1:3].to_string())
    return df.iloc[i]

def slice(df: pd.DataFrame, row_slice: list, column_slice: list):
    return df.loc[row_slice[0]:row_slice[1], column_slice[0]:column_slice[1]]

def unique_column(df: pd.DataFrame, column: str):
    return df[column].unique()

def filter_by_columns(df: pd.DataFrame, column: str, values_list: list, not_in: bool = False):
    if not_in:
        return df[~df[column].isin(values_list)]
    else:
        return df[df[column].isin(values_list)]

def select_dtypes(df: pd.DataFrame, dtypes_list: list, include_or_exclude: bool):
    if include_or_exclude:
        return df.select_dtypes(include=dtypes_list)
    else:
        return df.select_dtypes(exclude=dtypes_list)

def where_column_equals(df: pd.DataFrame, column: str, value):
    return df.loc[(df[column] == value)]

def where_column_doesnt_contain(df: pd.DataFrame, column: str, value: str):
    # print(df.loc[(df['Type 1'] == "Grass") & (~df['Name'].str.contains("v"))])
    return df.loc[~df[column].str.contains(value)]

def where_column_regex(df: pd.DataFrame, column: str, regex: str = "^pi[a-z]*"):
    # print(df.loc[df['Name'].str.contains("y|z", flags=re.I, regex=True)])
    # print(df.loc[df['Name'].str.contains("^pi[a-z]*", flags=re.I, regex=True)])
    return df.loc[df[column].str.contains(regex, flags=re.I, regex=True)]
#endregion

#region Sorting
def sort(df: pd.DataFrame, columns: list, asc_desc: list):
    # df.sort_values(['Type 1', 'Name'], ascending=[0,1], inplace=True)
    df.sort_values(columns, ascending=asc_desc, inplace=True)

def rearrange_columns(df: pd.DataFrame, new_columns_list):
    # cols = list(df.columns)
    # df = df[cols[0:4] + [cols[-1]] + cols[4:12]]
    return df[new_columns_list]

def reverse_rows(df: pd.DataFrame):
    reverse_df = df.loc[::-1]
    redefine_index(reverse_df)
    return reverse_df

def reverse_columns(df: pd.DataFrame):
    reverse_df = df.loc[:, ::1]
    return reverse_df
#endregion

#region Transform
def rename_columns(df: pd.DataFrame, from_to: dict):
    df.rename(from_to, axis='columns', inplace=True)

def prefix_columns(df: pd.DataFrame, string: str):
    df.add_prefix(str)

def suffix_columns(df: pd.DataFrame, string: str):
    df.add_suffix(str)

def drop_columns(df: pd.DataFrame, columns: list):
    df.drop(columns=columns, inplace=True, axis="columns")

def new_column_addition(df: pd.DataFrame, new_column: str, column1: str, column2: str):
    df[new_column] = df[column1] + df[column2]

def conditional_change_set_where(df: pd.DataFrame, column: str, if_value, set_column: str, to_value):
    # df.loc[df['Type 1'] == 'Fire', ['Type 1', 'Type 2']] = ['Flamer', 'Fiery']
    df.loc[df[column] == if_value, [set_column]] = [to_value]

def redefine_index(df: pd.DataFrame):
    df.reset_index(inplace=True, drop=True)

def string_to_numeric(df: pd.DataFrame):
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)

def append_dfs(df1: pd.DataFrame, df2: pd.DataFrame):
    return pd.concat([df1, df2], ignore_index=True)

def concact_dfs(df1: pd.DataFrame, df2: pd.DataFrame):
    return pd.concat([df1, df2], ignore_index=True, axis='columns')

def dropa_na_columns(df: pd.DataFrame, threshhold: float):
    return df.dropna(thresh=len(df) * threshhold, axis='columns')

def split_string_into_columns(df: pd.DataFrame, new_columns: list, column: str, separator: str):
    df[new_columns] = df[column].str.split(separator, expand=True)

def loop_iterate(df: pd.DataFrame):
    for index, row in df.iterrows():
        return (index, row)
#endregion

#region GroupBy
def group_by_mean(df: pd.DataFrame, group_by_column: str, sort_by_column: str):
    # df = df.groupby(['Type 1']).mean().sort_values('Defense', ascending=False)
    return df.groupby([group_by_column]).mean().sort_values(sort_by_column, ascending=False)

def group_by_count(df: pd.DataFrame, columns: list):
    # df['count'] = 1
    # df = df.groupby(['Type 1', 'Type 2']).count()['count']
    df['count'] = 1
    return df.groupby(columns).count()['count']
#endregion

#region Settings
def set_float_precision(precision: int):
    pd.set_option('display.float_format', '{:.xf}'.replace('x', precision).format)
#endregion
