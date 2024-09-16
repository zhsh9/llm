# Pandas

## load data

- read_csv()
- read_excel()
- read_sql()
- read_json()
- read_html()
- read_parquet()

## save data

- to_csv()
- to_excel()
- to_sql()
- to_json()
- to_html()
- to_parquet()

## preview data

- head()
- tail()
- info()
- describe()
- dtypes

## locate

- loc[]
- iloc[]
- query()

## preprocess

- dropna()
- fillna()
- drop()
- rename()
- astype()
- duplicated()
- drop_duplicated()

## transform

- groupby()
- pivot_table()
- merge()
- concat()
- apply()
- applymap()

## date

- to_datetime()
- resample()
- rolling()

## data concat

- pd.concat([df1, df2])
- pd.merge(df1, df2, on='key')
- df1.join(df2, how='left')
- df1.append(df2)

## index

- set_index('col')
- reset_index()
- swaplevel()
- stack()
- unstack()

## statistic

- unique()
- value_counts()
- sort_values()
- isna()
- sum()
- mean()
- max()
- min()
- median()
- std()
- var()
- count()
- agg()

## data type

- isnumeric()
- isdecimal()
- isalpha()
- isdigit()
- islower(), isupper(), istitle()

## data reformat

- melt()
- pivot()
- cut()
- qcut()

## string

- lower()
- upper()
- len()
- strip(), lstrip(), rstrip()
- contains()
- startswith()
- endswith()
- match()
- replace()
- split()
- join()
- cat()

## re

- extract()
- findall()
- replace()

## visual

- plot()
- plot.bar(), plot.barh()
- plot.hist()
- plot.box()
- plot.area()
- plot.pie()
- plot.scatter()

## expression

- eval()
- query()
