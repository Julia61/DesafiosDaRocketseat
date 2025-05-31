# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df_aluguel = pd.read_csv('./dataset/dataset_aluguel.csv')

# %%
df_aluguel.info()

# %%
df_aluguel.drop(columns=['id'], axis=1, inplace=True)

# %%
df_aluguel.columns = [
    'tamanho_metro_quadrado',
    'numero_de_quartos',
    'idade_da_casa',
    'contem_garagem',
    'localizacao_periferia',
    'localizacao_suburbio',
    'valor_do_aluguel'
]

# %% [markdown]
# ### EDA

# %%
df_aluguel.head(10)

# %%
df_aluguel.isna().sum()

# %%
df_aluguel.describe()

# %%
sns.boxplot(data=df_aluguel, x='tamanho_metro_quadrado')

# %%
sns.boxplot(data=df_aluguel, x='numero_de_quartos')

# %%
sns.boxplot(data=df_aluguel, x='idade_da_casa')


# %%
sns.boxplot(data=df_aluguel, x='valor_do_aluguel')


# %%
sns.scatterplot(data=df_aluguel, x='tamanho_metro_quadrado', y='valor_do_aluguel')

# %%
sns.scatterplot(data=df_aluguel, x='numero_de_quartos', y='valor_do_aluguel')

# %%
plt.figure(figsize=(15,6))
sns.heatmap(df_aluguel.corr(), vmin=-1, vmax=1, annot=True)

# %%

sns.heatmap(df_aluguel.corr()[['valor_do_aluguel']].sort_values(by='valor_do_aluguel', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')

# %% [markdown]
# ### Treinar Modelo

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# %%
X = df_aluguel.drop(columns='valor_do_aluguel', axis=1)
y = df_aluguel['valor_do_aluguel']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=51)

# %%
colunas_numericas = ['tamanho_metro_quadrado','numero_de_quartos', 'idade_da_casa', 'contem_garagem']
colunas_boolean = ['localizacao_periferia', 'localizacao_suburbio']

# %%
tranformer_numericas = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# %%
tranformer_boolean = Pipeline(steps=[
    ('bool_to_int', FunctionTransformer(lambda x: x.astype(int)))
])

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', tranformer_numericas,colunas_numericas),
        ('cat', tranformer_boolean,colunas_boolean)
    ]
)

# %%
model_reg = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])

# %%
model_reg.fit(X_train, y_train)

# %%
y_pred = model_reg.predict(X_test)

# %%
predicao = {
    'tamanho_metro_quadrado': 292.60714596148742,
    'numero_de_quartos': 6,
    'idade_da_casa': 36.7608059620386,
    'contem_garagem': 1,
    'localizacao_periferia': 1,
    'localizacao_suburbio': 0
}
sample_df = pd.DataFrame(predicao, index=[1])

# %%
model_reg.predict(sample_df)

# %% [markdown]
# ## Métricas

# %% [markdown]
# #### MAE

# %%
mean_absolute_error(y_test, y_pred)

# %% [markdown]
# #### RMSE

# %%
root_mean_squared_error(y_test,y_pred)

# %% [markdown]
# #### R²

# %%
r2_score(y_test, y_pred)

# %% [markdown]
# ## Análise de Residuos

# %%
residuos = y_test - y_pred

# %%
from scipy.stats import zscore
residuos_std = zscore(residuos)

# %%
sns.scatterplot(x=y_pred, y=residuos_std)
plt.axhline(y=0)
plt.axhline(y=-2)
plt.axhline(y=2)

# %%
import pingouin as pg
plt.figure(figsize=(14, 8))
pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Quantis Teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.show()

# %%
from scipy.stats import shapiro, kstest, anderson
stat_shapiro, p_value_shapiro = shapiro(residuos)
print("Estatistica do Teste: {} e P-Value: {}".format(stat_shapiro,p_value_shapiro))

# %%
stat_ks, p_value_ks = kstest(residuos_std, 'norm')
print("Estatistica do Teste: {} e P-Value: {}".format(stat_ks,p_value_ks))

# %%
modelo = model_reg.named_steps['regressor']

# %%

colunas_numericas_transformadas = colunas_numericas
colunas_boolean_transformadas = colunas_boolean

nomes_finais = colunas_numericas_transformadas + colunas_boolean_transformadas


# %%
for nome, coef in zip(nomes_finais, modelo.coef_):
    print(f'{nome}: {coef:.4f}')


# %% [markdown]
# ## Limitações do Modelo e Possíveis Melhorias

# %% [markdown]
# Uma das limitações identificadas no dataset foi que a variável idade_da_casa está representada como float, o que pode não fazer sentido prático, já que a idade de uma casa geralmente é expressa em números inteiros (anos completos). Essa imprecisão pode introduzir ruídos no modelo, influenciando negativamente na interpretação e na performance.
# 
# Além disso, apesar do modelo ter apresentado um R² muito alto (0.99) — indicando que ele explica bem a variação dos dados —, é importante ficar atento à possibilidade de overfitting. Isso pode ocorrer caso o modelo esteja ajustado demais aos dados de treino e perca a capacidade de generalizar.
# 
# O teste de normalidade dos resíduos com o Kolmogorov-Smirnov (KS) indicou que os resíduos não seguem perfeitamente uma distribuição normal, o que pode violar um dos pressupostos da regressão linear e afetar a confiabilidade de algumas inferências.
# 
# ####  Pontos fortes
# 
# - O modelo conseguiu prever os valores de aluguel com baixa margem de erro (MAE ≈ 40).
# 
# - Resíduos razoavelmente bem distribuídos, como mostrado no gráfico de resíduos padronizados.
# 
# - Coeficientes interpretáveis que ajudam a entender o impacto de cada variável.
# 
# 
# #### Possíveis melhorias
# 
# - Arredondar a idade da casa para números inteiros.
# 
# - Adicionar novas variáveis, como proximidade a transporte público, comércio, ou condição do imóvel.
# 
# - Avaliar outliers com mais profundidade, especialmente nas variáveis tamanho_metro_quadrado e valor_do_aluguel, para reduzir distorções.


