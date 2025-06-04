# %%
import pandas as pd
import pingouin as pg
import plotly.express as px 
import plotly.figure_factory as ff
import matplotlib.pyplot as plt


# Machine Learning
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay

##Otimização de Hiperpâmetros
import optuna

# %%
df_transacoes = pd.read_csv('./datasets/transacoes_fraude.csv')

# %% [markdown]
# ### EDA

# %%
df_transacoes.head(10)

# %%
df_transacoes.info()

# %%
df_transacoes['Horario da Transacao'] = pd.to_datetime(df_transacoes['Horario da Transacao'])


# %%
df_transacoes['Tipo de Transacao'].unique()

# %%
contagem_target = df_transacoes.value_counts('Classe')
contagem_target

# %%
px.bar(contagem_target, color=contagem_target.index)

# %%
percentual_target = contagem_target / len(df_transacoes) * 100
px.bar(percentual_target, color=percentual_target.index, )

# %%
percentual_tipo_transasao = df_transacoes.value_counts('Tipo de Transacao') / len(df_transacoes) * 100
px.bar(percentual_tipo_transasao, color=percentual_tipo_transasao.index)

# %%
lista = ['Saque', 'PIX', 'Débito', 'Crédito']

# %%
crosstab_tipo_transasao = pd.crosstab(df_transacoes['Classe'], df_transacoes['Tipo de Transacao'], margins=True).reset_index()

tabela_tipo_transasao = ff.create_table(crosstab_tipo_transasao)

tabela_tipo_transasao.show()

# %%
valor_esperado, valor_observado, estatisticas = pg.chi2_independence(df_transacoes, 'Classe', 'Tipo de Transacao')

# %%
valor_esperado

# %%
valor_observado

# %%
estatisticas.round(5)

# %% [markdown]
# As variáveis Tipo de Transacao e Classe não são independentes. Qui-Quadrado (p-value = 0.034)
# As variáveis Valor da Transacao e Classe são independentes. Qui-Quadrado (p-value = 0.545)
# As variáveis Valor Anterior a Transacao e Classe são independentes. Qui-Quadrado (p-value = 0.491)
# As variáveis Horario da Transacao e Classe são independentes. Qui-Quadrado (p-value = 0.495)

# %% [markdown]
# ### Treinamento do Modelo

# %%
df_transacoes.drop(columns=['Cliente'], axis=1, inplace=True)

# %%
X = df_transacoes.drop(columns=['Classe'])
y = df_transacoes['Classe']

# %%
categorical_features = ['Tipo de Transacao']

categorical_tranformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_tranformer, categorical_features)
    ]
)

dt_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', DecisionTreeClassifier())])

# %% [markdown]
# ### Validação Cruzada

# %%
cv_foldds = StratifiedKFold(n_splits=3, shuffle=True, random_state=51)
metrics_result = cross_validate(dt_model, X, y, cv=cv_foldds, scoring=['accuracy'],
return_estimator = True)

# %%
metrics_result

# %%
metrics_result['test_accuracy'].mean()

# %% [markdown]
# ### Métricas
# 

# %%
y_pred = cross_val_predict(dt_model, X, y, cv=cv_foldds)

# %%
classification_report_str = classification_report(y, y_pred)

print(f'Relatório de classificação:\n{classification_report_str}')

# %%
confusion_matrix_modelo = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix_modelo)
disp.plot()

# %% [markdown]
# ### Tuning de Hiperparâmetros

# %%
def decisiontree_optuna(trial):

    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_depth = trial.suggest_int('max_depth', 2, 8)

    dt_model.set_params(classifier__min_samples_leaf = min_samples_leaf)
    dt_model.set_params(classifier__max_depth= max_depth)

    scores = cross_val_score(dt_model, X, y, cv=cv_foldds, scoring='accuracy')

    return scores.mean()

# %%
estudo_decisiontree = optuna.create_study(direction='maximize')
estudo_decisiontree.optimize(decisiontree_optuna, n_trials=200)

# %%
print(f'Melhor acurácia: {estudo_decisiontree.best_value}')
print(f'Melhores parâmetros: {estudo_decisiontree.best_params}')

# %% [markdown]
# ### Visualizar Árvore

# %%
X_train_tree = X.copy()
X_train_tree['Tipo de Transacao_label'] = X_train_tree['Tipo de Transacao'].astype('category').cat.codes
X_train_tree.drop(columns=['Tipo de Transacao'], axis=1, inplace=True)
X_train_tree.rename(columns={'Tipo de Transacao_label' : 'Tipo de Transacao'}, inplace=True)
X_train_tree.head(10)

# %%
X_train_tree = X_train_tree.select_dtypes(exclude=['datetime64[ns]', 'datetime64'])


# %%
clf_decisiontree = DecisionTreeClassifier(min_samples_leaf=estudo_decisiontree.best_params['min_samples_leaf'], max_depth=estudo_decisiontree.best_params['max_depth'])

y_train_tree = y.copy()

clf_decisiontree.fit(X_train_tree, y_train_tree)

# %%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10), dpi=600)

plot_tree(clf_decisiontree,
          feature_names=X_train_tree.columns.to_numpy(),
          filled=True)

# %% [markdown]
# O modelo de árvore de decisão conseguiu detectar 1 % de não fraude (Recall = 1.00), o que não é muito positivo. E ele apresentou 0 % de falsos positivos, impactando a experiência de clientes legítimos. Isso mostra que a árvore de decisão não é uma melhor escolha


