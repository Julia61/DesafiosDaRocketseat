# %%
import pandas as pd

# %%
df_netflix = pd.read_csv('netflix_daily_top_10.csv')

# %%
df_netflix.head(5)

# %%
df_netflix.info()

# %%
df_netflix['As of'] = pd.to_datetime(df_netflix['As of'])

# %%
df_netflix.info()

# %%
df_netflix['As of'].min()

# %%
df_netflix['As of'].max()

# %%
df_netflix.shape

# %%
df_netflix.isnull().sum()

# %%
df_netflix.fillna(0, inplace=True)

# %%
df_netflix['Rank'].plot.box()

# %% [markdown]
# - Os dados de Rank estão bem distribuídos entre 1 e 10
# 
# - A maior parte dos títulos ficam entre 3 e 8
# 
# - Não tem outliers porque o Rank só vai até 10, naturalmente

# %%
df_netflix['Days In Top 10'].plot.box()

# %% [markdown]
# - A maioria dos títulos da Netflix passa bem pouco tempo no Top 10.
# 
# - Existem alguns poucos títulos extremamente populares que dominam o Top 10 por centenas de dias.
# 
# - Esses casos são exceções e aparecem como outliers no boxplot.


