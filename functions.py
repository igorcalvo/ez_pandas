import pandas as pd

def print_df(df: pd.DataFrame):
    print(df.to_string())

def print_beginning(df: pd.DataFrame, rows: int):
    print(df.head(rows).to_string())

def print_ending(df: pd.DataFrame, rows: int):
    print(df.tail(rows).to_string())

def get_headers(df: pd.DataFrame):
    return df.columns

def get_value(df: pd.DataFrame, row: int, column: int):
    return df.iloc[row,column]

def function(df: pd.DataFrame):
    pass