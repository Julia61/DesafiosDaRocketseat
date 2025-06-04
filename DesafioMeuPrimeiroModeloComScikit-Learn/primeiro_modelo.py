# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
dados_vendas = {
    'mes': ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'],
    'vendas': [2000, 2200, 2300, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300]
}

# %%
df_vendas = pd.DataFrame.from_dict(dados_vendas)

# %%
df_vendas

# %%
df_vendas['mes_num'] = df_vendas['mes'].apply(lambda x: df_vendas['mes'].tolist().index(x) + 1)

# %%
# Importar bibliotecas so sklearn
from sklearn.linear_model import LinearRegression


# %%
X = df_vendas[['mes_num']]
y = df_vendas[['vendas']]

# %%
X_train = X[:-1]
y_train = y[:-1]

# %%
X_teste = X[-1:]
y_teste = y[-1:]

# %%
model = LinearRegression().fit(X_train, y_train)

# %%
y_pred = model.predict(X_teste)

# %%
y_teste.values[0]


# %%
y_pred[0]

# %%
df_vendas.hist(bins=10, figsize=(10,6), grid=True)
plt.suptitle('Histogramas de Vendas e Meses')
plt.show()

# %%

plt.scatter(X_teste, y_teste, color='green', label='Valor Real (Dezembro)', s=100)


plt.scatter(X_teste, y_pred, color='red', label='Valor Previsto (Dezembro)', s=100, marker='x')


plt.plot(X, model.predict(X), color='blue', label='Linha de tendência')

plt.title('Previsão de Vendas para Dezembro')
plt.xlabel('Mês (numérico)')
plt.ylabel('Vendas')
plt.legend()
plt.grid(True)
plt.show()

# %%



