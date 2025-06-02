# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ### Carregamento de Dados

# %%
df_receita = pd.read_csv('./datasets/sales_data.csv')

# %%
df_receita.info()

# %%
df_receita_eda = df_receita.copy()

# %% [markdown]
# ### Exploração dos Dados

# %%
df_receita_eda.head(10)

# %%
df_receita_eda.describe()

# %%
df_receita_eda.isna().sum()

# %%
sns.boxplot(data=df_receita_eda, x='tempo_de_experiencia')

# %%
sns.boxplot(data=df_receita_eda, x='numero_de_vendas')

# %%
sns.boxplot(data=df_receita_eda, x='fator_sazonal')

# %%
sns.boxplot(data=df_receita_eda, x='receita_em_reais')

# %%
sns.scatterplot(data=df_receita_eda, x='tempo_de_experiencia', y='receita_em_reais')

# %%
sns.scatterplot(data=df_receita_eda, x='numero_de_vendas', y='receita_em_reais')

# %%
sns.scatterplot(data=df_receita_eda, x='fator_sazonal', y='receita_em_reais')

# %%
sns.pairplot(df_receita_eda)

# %%
plt.figure(figsize=(15,6))
sns.heatmap(df_receita_eda.corr('spearman',), vmin=-1, vmax=1, annot=True)

# %%
sns.heatmap(df_receita_eda.corr('spearman')[['receita_em_reais']].sort_values(by='receita_em_reais', ascending=False), vmin=-1,vmax=1,annot=True, cmap='BrBG')

# %% [markdown]
# ### Treinar Modelo Linear

# %%
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error

import numpy as np

# %%
X = df_receita.drop(columns='receita_em_reais', axis=1)
y = df_receita['receita_em_reais']

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=51)

# %%
kf.split(X)

# %%
colunas_numericas = ['tempo_de_experiencia', 'numero_de_vendas', 'fator_sazonal']

tranformer_numericas = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', tranformer_numericas, colunas_numericas)
    ]
)

model_reg = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])

rmse_scores_fold_train = []
rmse_scores_fold_test = []

r2score_fold_test = []

residuos = []

mse_fold_test = []

y_pred_total = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model_reg.fit(X_train, y_train)

    y_train_pred = model_reg.predict(X_train)
    y_test_pred = model_reg.predict(X_test)

    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    r2score_test = r2_score(y_test, y_test_pred)
    residuos_test = np.array(y_test - y_test_pred)

    rmse_scores_fold_train.append(rmse_train)
    rmse_scores_fold_test.append(rmse_test)
    mse_fold_test.append(mse_test)
    r2score_fold_test.append(r2score_test)
    residuos.append(residuos_test)
    y_pred_total.append(y_test_pred)

rmse_train_final = np.mean(rmse_scores_fold_train)    
rmse_test_final = np.mean(rmse_scores_fold_test)
mse_test_final = np.mean(mse_fold_test)   
r2score_test_final = np.mean(r2score_fold_test)
percentual_rmse_final = ((rmse_test_final - rmse_train_final) / rmse_train_final) * 100
residuos = np.array(residuos).reshape(-1)
y_pred_total = np.array(y_pred_total).reshape(-1)




# %% [markdown]
# ### Análise de Métricas - Modelo Linear

# %%

print(f'RMSE Treino: {rmse_train_final}')
print(f'RMSE Teste: {rmse_test_final}')
print(f'% Dif. RMSE Treino e Teste: {percentual_rmse_final}')
print(f'R2Score Teste: {r2score_test_final}')
print(f'MSE: {mse_test_final}')

# %% [markdown]
# ### Análise de Resíduos - Modelo Linear

# %%
from scipy.stats import zscore
residuos_std = zscore(residuos)

# %%
sns.scatterplot(x=y_pred_total, y=residuos_std)
plt.axhline(y=0)
plt.axhline(y=-2)
plt.axhline(y=2)

# %%
import pingouin as pg
plt.figure(figsize=(14,8))
pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Quantis Teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.show()

# %%
from scipy.stats import shapiro, kstest
from statsmodels.stats.diagnostic import lilliefors
stat_shapiro, p_value_shapiro = shapiro(residuos)
print(f'Estat. Teste {stat_shapiro} e P-Value {p_value_shapiro}')

# %%
stat_ks, p_value_ks = kstest(residuos, 'norm')
print(f'Estat. Teste {stat_ks} e P-Value {p_value_ks}')

# %%
stat_ll, p_value_ll = lilliefors(residuos, dist='norm', pvalmethod='table')
print(f'Estat. Teste {stat_ll} e P-Value {p_value_ll}')

# %% [markdown]
# ### Treinar Modelo Polinomial

# %%
graus_polynomial = [2]

rmse_train_values = []
rmse_test_values = []
percentual_rmse_values = []
r2score_test_values = []

kf = KFold(n_splits=5, shuffle=True, random_state=51)

for grau in graus_polynomial:
    colunas_numericas = ['tempo_de_experiencia', 'numero_de_vendas', 'fator_sazonal']

    tranformer_numericas = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', tranformer_numericas, colunas_numericas)
        ]
    )

    poly_feat = PolynomialFeatures(degree=grau, include_bias=False)

    model_poly = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('poly_features', poly_feat),
                                 ('regressor', LinearRegression())])
    
    rmse_scores_fold_train = []
    rmse_scores_fold_test = []

    r2score_fold_test = []

    residuos = []

    y_pred_total = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_poly.fit(X_train, y_train)

        y_train_pred = model_poly.predict(X_train)
        y_test_pred = model_poly.predict(X_test)

        rmse_train = root_mean_squared_error(y_train, y_train_pred)
        rmse_test = root_mean_squared_error(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        r2score_test = r2_score(y_test, y_test_pred)
        residuos_test = np.array(y_test - y_test_pred)

        rmse_scores_fold_train.append(rmse_train)
        rmse_scores_fold_test.append(rmse_test)
        mse_fold_test.append(mse_test)
        r2score_fold_test.append(r2score_test)
        residuos.append(residuos_test)
        y_pred_total.append(y_test_pred)

    rmse_train_final = np.mean(rmse_scores_fold_train)    
    rmse_test_final = np.mean(rmse_scores_fold_test)
    mse_test_final = np.mean(mse_fold_test)   
    r2score_test_final = np.mean(r2score_fold_test)
    percentual_rmse_final = ((rmse_test_final - rmse_train_final) / rmse_train_final) * 100
    residuos = np.array(residuos).reshape(-1)
    y_pred_total = np.array(y_pred_total).reshape(-1)

    rmse_train_values.append(rmse_train_final)
    rmse_test_values.append(rmse_test_final)
    r2score_test_values.append(r2score_test_final)
    percentual_rmse_values.append(percentual_rmse_final)


# %%
plt.figure(figsize=(12,8))
plt.plot(graus_polynomial, rmse_train_values, label='RMSE (Treino)')
plt.plot(graus_polynomial, rmse_test_values, label='RMSE (Teste)')
plt.xlabel('Grau do Polinômio')
plt.ylabel('RMSE')
plt.title('RMSE por Grau do Polinômio')
plt.legend()
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(12,8))
plt.plot(graus_polynomial, percentual_rmse_values, label='%Dif RMSE Treino e Teste')
plt.xlabel('Grau do Polinômio')
plt.ylabel('%Dif RMSE')
plt.title('%Dif RMSE por Grau do Polinômio')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### Análise de Métricas - Modelo Polinomial

# %%
print(f'RMSE Treino: {rmse_train_final}')
print(f'RMSE Teste: {rmse_test_final}')
print(f'% Dif. RMSE Treino e Teste: {percentual_rmse_final}')
print(f'R2Score Teste: {r2score_test_final}')
print(f'MSE: {mse_test_final}')

# %%
from scipy.stats import zscore
residuos_std = zscore(residuos)

# %%
sns.scatterplot(x=y_pred_total, y=residuos_std)
plt.axhline(y=0)
plt.axhline(y=-2)
plt.axhline(y=2)

# %%
import pingouin as pg
plt.figure(figsize=(14,8))
pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Quantis Teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.show()

# %%
from scipy.stats import shapiro, kstest
from statsmodels.stats.diagnostic import lilliefors
stat_shapiro, p_value_shapiro = shapiro(residuos)
print(f'Estat. Teste {stat_shapiro} e P-Value {p_value_shapiro}')

# %%

stat_ks, p_value_ks = kstest(residuos, 'norm')
print(f'Estat. Teste {stat_ks} e P-Value {p_value_ks}')


# %%

stat_ll, p_value_ll = lilliefors(residuos, dist='norm', pvalmethod='table')
print(f'Estat. Teste {stat_ll} e P-Value {p_value_ll}')


# %%
input_features = {
    'tempo_de_experiencia': 50,
    'numero_de_vendas': 11,
    'fator_sazonal': 5
}

pred_df = pd.DataFrame(input_features, index=[1])

# %%
model_poly.predict(pred_df)

# %%
import joblib

# %%
joblib.dump(model_poly, './modelo_receita.pkl')


