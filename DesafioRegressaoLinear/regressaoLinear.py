
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro, kstest, probplot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

df_irigacao = pd.read_csv('./dataset/dados_de_irrigacao.csv')

df_irigacao.info()

df_irigacao.head(10)

df_irigacao.tail(10)

df_irigacao.describe()

sns.scatterplot(data=df_irigacao, x='Horas de Irrigação', y='Área Irrigada')

sns.boxplot(df_irigacao, y='Área Irrigada')

sns.boxplot(df_irigacao, y='Horas de Irrigação')

sns.heatmap(df_irigacao.corr('pearson'), annot=True)

sns.heatmap(df_irigacao.corr('spearman'), annot=True)

sns.displot(df_irigacao, x='Área Irrigada')

sns.displot(df_irigacao, x='Horas de Irrigação')

X = df_irigacao['Horas de Irrigação'].values.reshape(-1,1)
y = df_irigacao['Área Irrigada'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)

reg_model = LinearRegression()

reg_model.fit(X_train, y_train)

print("A equação da reta é y = {:4f}x + {:4f}".format(reg_model.coef_[0][0], reg_model.intercept_[0]))


y_pred = reg_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error


mean_squared_error(y_test, y_pred)

mean_absolute_error(y_test,y_pred)

residuos = y_test - y_pred

from scipy.stats import zscore
residuos_std = zscore(residuos)

sns.scatterplot(x=y_pred.reshape(-1), y=residuos_std.reshape(-1))
plt.axhline(y=0)

import pingouin as pg
pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Horas de Irrigação')
plt.ylabel('Residuos na escala padrão')
plt.show()

stat_shapiro, p_valor_shapiro = shapiro(residuos.reshape(-1))
print("Estatística do teste: {} e P-Valor: {}".format(stat_shapiro, p_valor_shapiro))

stat_ks, p_valor_ks = kstest(residuos.reshape(-1), 'norm')
print("Estatística do teste: {} e P-Valor: {}". format(stat_ks, p_valor_ks))

reg_model.predict([[15]])

reg_model.predict([[75.50]])

reg_model.predict([[150]])

reg_model.predict([[224.50]])

reg_model.predict([[299]])

reg_model.predict([[2]])


