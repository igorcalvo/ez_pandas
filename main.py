# https://www.youtube.com/watch?v=hl-TGI4550M&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=1

import pandas as pd
import re
from functions import *

file_name = 'Pokemon.csv'
df = pd.read_csv(file_name)
# print(df.head(3).to_string())

# prints
# print_df(df)
# print_beginning(df, 5)
# print_ending(df, 3)

# headers
# print(get_headers(df))
# print(df['Name'][:5])
# print(df[['Attack', 'Defense']])

# rows
# print(df[0:5].to_string())
# print(df.iloc[5].to_string())
# print(df.iloc[1:3].to_string())

# header and row
# print(df.iloc[2,2])
# print(get_value(df, 2, 2))

# looping
# for index, row in df.iterrows():
#     print(index, row['Type 1'])

# where
# print(df.loc[(df['Type 1'] == "Grass") & (~df['Name'].str.contains("v"))])
# print(df.loc[df['Name'].str.contains("y|z", flags=re.I, regex=True)])
# print(df.loc[df['Name'].str.contains("^pi[a-z]*", flags=re.I, regex=True)])
# df.reset_index(inplace=True, drop=True)

# sort / describe
# print(df.describe().to_string())
# df.sort_values(['Type 1', 'Name'], ascending=[0,1], inplace=True)
# print(df)

# operating
# print(get_headers(df))
# df['Stats'] = df['Attack'] + df['Defense']
# df = df.drop(columns=['Sp. Atk', 'Sp. Def'])

# sorting
# cols = list(df.columns)
# df = df[cols[0:4] + [cols[-1]] + cols[4:12]]
# print_df(df)

# saving
# df.to_csv("filename.csv", index=False)

# conditional changes
# df.loc[df['Type 1'] == 'Fire', ['Type 1', 'Type 2']] = ['Flamer', 'Fiery']
# print_df(df)

# groupby
# df = df.groupby(['Type 1']).mean().sort_values('Defense', ascending=False)
# print_df(df)
# ---
# df['count'] = 1
# df = df.groupby(['Type 1', 'Type 2']).count()['count']
# print_df(df)


# reading and grouping by chuncks
# new_df = pd.DataFrame(columns = df.columns)
# # chunksize = rows
# for df in pd.read_csv(file_name, chunksize=5):
#     results = df.groupby(['Type 1']).count()
#     new_df = pd.concat([new_df, result])