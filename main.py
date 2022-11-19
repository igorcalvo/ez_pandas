from functions import *

file_name = 'Pokemon.csv'
df = read_csv(file_name, ',')
df2 = read_csv(file_name, ',')
# drop_columns(df, ['Generation'])