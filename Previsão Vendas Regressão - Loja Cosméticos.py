#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pysal==1.14.4')


# In[2]:


get_ipython().system('pip install feature-engine')


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson

from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale


get_ipython().run_line_magic('matplotlib', 'inline')
import scipy as sp
import statsmodels.formula.api as sm
import statsmodels.api as sm

from scipy import stats

import statsmodels.api as sm



from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

pd.options.display.max_rows = 999
pd.set_option("display.float_format", lambda x: "%.2f" %x)



import plotly.express as px
import plotly.graph_objects as go

import datetime as dt


# ## Previsão Vendas usando Regressão 

# ## <center> 1. Banco de Dados </center>

# In[4]:


dados = pd.read_excel("C:/Users/aline.nunes/Documents/ProjetoSupervisionado/Modelagem Matriz-20221118T223710Z-001/Modelagem Matriz/Unidade Matriz/DadosLojaCosmeticos.xlsx")

colunas = ['dia', 'mês', 'ano', 'Dia_da_semana', 'Num_Semana_mes', 'feriados', 'pagamento', 'Vale']
for i in range(len(colunas)):
    dados[colunas[i]] = dados[colunas[i]].astype(int)
dados = dados[dados['Data'] < '2021-04-01']

dados.head(5)


# ## Análise Exploratória

# In[5]:


fig1 = px.line(dados[dados['Vendas'] != 0.00].groupby(["Data", "Dia_da_Semana"])["Vendas"].sum().reset_index(), x="Data", y="Vendas", color= "Dia_da_Semana" ,template='plotly_white' ,labels={
    'Vendas': 'Vendas (R$)'})
fig1.update_layout(title={
    'text' : 'Vendas diárias do ano 2018 até março 2021 por dia da semana'})

fig2 = px.box(dados[dados['Vendas'] != 0.00].groupby(["Data", "Dia_da_Semana", "Dia_da_semana"])["Vendas"].sum().reset_index().sort_values(['Dia_da_semana']), x="Dia_da_Semana", y="Vendas", template='plotly_white',labels={
    'Vendas': 'Vendas (R$)', 'Dia_da_Semana': 'Dia da semana'})
fig2.update_layout(title={
    'text' : 'Distribuição das vendas do ano 2018 até março 2021 por dia da semana'})


fig1.show()
fig2.show()


# ## <center> 2. Adição Variáveis Dummies </center>

#    ### 2.1 Matriz Dummy Numeração da Semana do Mês, dia da semana
# 
# É considerado como início da semana, domingo. 
# <p>Referência: Quinta semana do mês
#     
# <p>Referêcia: Segunda-Feira

# In[6]:


dados = pd.get_dummies(dados, columns=["Num_Semana_mes",  "Dia_da_Semana"])


# ### 2.3 Matriz Dummy dos Outliers

# In[7]:


dados['Outlier_BlackFriday'] = np.where((dados['Data'] == '2019-11-29') |  (dados['Data'] == '2020-11-27'), 1, 0 )

dados['Outlier_SemanaSanta'] = np.where((dados['Data'] == '2018-03-31') |  (dados['Data'] == '2019-04-18') |
                                        (dados['Data'] == '2019-04-20') |  (dados['Data'] == '2020-04-09') |
                                        (dados['Data'] == '2020-04-11') |  (dados['Data'] == '2021-04-01') |
                                        (dados['Data'] == '2021-04-03'), 1, 0 )

dados['Vale'] = np.where((dados['dia'] == 19) & (dados['Dia_da_semana'] == 6) | 
                        (dados['dia'] == 18) & (dados['Dia_da_semana'] == 6) |
                        (dados['dia'] == 20) & (dados['Dia_da_semana'] > 1) & (dados['Dia_da_semana'] <= 6), 1, 0 )   


# In[8]:


dados = dados[dados['Vendas'] > 1]


# ## Inserindo Informações Publicas

# In[9]:


def consulta_bc(codigo_bc):
    url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bc)
    df = pd.read_json(url)
    df['data'] = pd.to_datetime(df['data'], dayfirst = True)
    df.set_index('data', inplace = True)
    return df.loc['2018-01-01' : '2021-03-31']

pib = consulta_bc(4380)
tx_selic = consulta_bc(4390)
desemprego = consulta_bc(24369) 
m1 = consulta_bc(27791)
igpm = consulta_bc(189)

data = [[pib, tx_selic, desemprego, igpm]]

index = pd.date_range(start = '2018-01-01', end = '2021-03-31', freq = "D")

df = pd.DataFrame(data, columns = [['Pib', 'Selic', 'Desemprego', 'IGP-M']], index = index)

data = [pib, tx_selic, desemprego, igpm]
df = pd.concat(data, axis=1)
df.columns = ['Pib', 'Selic', 'Desemprego', 'IGP-M']

df = df.reset_index()

df['data'] = pd.to_datetime(df['data'])
df['data'] = df['data'].dt.strftime('%Y-%m')


dados['data'] = pd.to_datetime(dados['Data'])
dados['data'] = dados['Data'].dt.strftime('%Y-%m')

dados_publicos = pd.merge(dados, df, how='left', on=['data'])
dados_publicos['mês_ano'] = dados_publicos['Data'].dt.strftime('%Y-%m')


# ### Correlação Variáveis continuas

# In[10]:


def plot_correlation_matrix(correlograma, remove_diagonal=False, remove_triangle=True, **kwargs):
    corr = correlograma.corr()
    # Apply mask
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = remove_triangle
    mask[np.eye(mask.shape[0], dtype=bool)] = remove_diagonal
    # Plot
    plt.figure(figsize=(13,5))
    sns.heatmap(corr, mask=mask, **kwargs, annot = True, cmap= 'Blues')
    plt.suptitle('Correlograma', fontsize=1)
    
    plt.show()

plot_correlation_matrix(dados_publicos[['Vendas','Pib', 'Selic','Desemprego', 'IGP-M']])


# ### <center> 3. Dados Prontos para serem usados <center>

# In[11]:


## Colocando Data como índice


#dadosProntos = dados[(dados['ano'] != '2018') & (dados['Vendas'] != 0.00)] 

dadosProntos = dados_publicos[(dados_publicos['ano'] != '2018') & (dados_publicos['Vendas'] != 0.00)] 
dadosProntos.set_index('Data', inplace = True)


# ## <center> 4. Modelagem </center>

# ### 4.1 Modelagem com os dados brutos
# 
# Não inclui no conjunto de dados as datas faltantes. Foram consideradas como variáveis independentes:
# 
# * feriados
# * pagamento
# * PrimeiraSemMes
# * TerceiraSemMes
# * QuartaSemMes
# * QuintaSemMes
# * SextaSemMes
# * domingo
# * terça
# * quarta
# * quinta
# * sexta
# * sabado
# * Outlier_DecTerceiro
# * Outlier_SemSanta

# #### 4.1 3. Separando em treino e teste

# In[13]:


dados.columns


# In[14]:


t_treino_apartir19 = dadosProntos.Vendas[dadosProntos.Vendas.index <'2021-01-01'].index.values
t_teste_apartir19 = dadosProntos.Vendas[(dadosProntos.Vendas.index >= '2021-01-01') & (dadosProntos.Vendas.index < '2021-04-01')].index.values
Y_treino_apartir19 = dadosProntos.Vendas[dadosProntos.Vendas.index < '2021-01-01'].values
Y_teste_apartir19 = dadosProntos.Vendas[(dadosProntos.Vendas.index >= '2021-01-01') & (dadosProntos.Vendas.index < '2021-04-01')].values

x_real3 = dadosProntos.drop(['dia', 'mês', 'ano','Dia_da_semana', 'Vendas', 'pagamento', 'Vale', 'mês_ano', 'data', 'feriados', 'Dia_da_Semana_Domingo', 'Num_Semana_mes_1', 'IGP-M', 'Pib', 'Selic', 'DiasNaoUteis',
       'Dia_Úteis'], axis = 1)
#x_real3 = dadosProntos.drop(['dia', 'mês', 'ano','Dia_da_semana', 'Vendas', 'pagamento', 'Vale', 'mês_ano', 'data', 'feriados', 'Dia_da_Semana_Domingo', 'Num_Semana_mes_1', 'IGP-M', 'Pib', 'Selic'], axis =1) melhor
#x_real3 = dadosProntos.drop(['dia', 'mês', 'ano', 'Dia_da_semana', 'DiasNaoUteis',
       #'Dia_Úteis', 'Vendas', 'mês_ano', 'data', 'Num_Semana_mes_5', 'Dia_da_Semana_Segunda', 'feriados', 'pagamento', 'Vale', 'IGP-M'], axis = 1) Modelo2
# x_real3 = dadosProntos.drop(['dia', 'mês', 'ano', 'Dia_da_semana', 'DiasNaoUteis',
       #'Dia_Úteis', 'Vendas', 'mês_ano', 'data', 'Num_Semana_mes_5', 'Dia_da_Semana_Segunda', 'IGP-M', 'Pib', 'Selic','Desemprego'], axis = 1)modelo3 desconsideera todas as variaveis publicas
t_treino_exog = x_real3[x_real3.index < '2021-01-01'].index.values
t_teste_exog = x_real3[(x_real3.index >= '2021-01-01') & (x_real3.index < '2021-04-01')].index.values
X_treino_exog = x_real3[x_real3.index < '2021-01-01'].values
X_teste_exog = x_real3[(x_real3.index >= '2021-01-01') & (x_real3.index < '2021-04-01')].values


# #### 4.1 4. Modelagem 

# In[15]:


modelPond = LinearRegression().fit(X_treino_exog, Y_treino_apartir19)

print('R2:', modelPond.score(X_treino_exog, Y_treino_apartir19)) 


# In[16]:


previsaoPond_X = modelPond.predict(X_teste_exog)
residuo = Y_teste_apartir19 - previsaoPond_X

#previsaoPond = modelPond.predict(X_teste_exog)


# In[17]:


#### Acuracia do modelo Base teste
from sklearn.metrics import mean_absolute_error, mean_squared_error

R2 = r2_score(Y_teste_apartir19, previsaoPond_X )
MAE = mean_absolute_error(Y_teste_apartir19, previsaoPond_X )
RMSE = np.sqrt(mean_squared_error(Y_teste_apartir19, previsaoPond_X ))
print("MAE = {:0.2f}".format(MAE))
print("RMSE = {:0.2f}".format(RMSE))
print("R2= {:0.2f}".format(R2))


# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def scatterPlot(actual, predicted):
    plt.subplots(figsize=(10,4), sharex=True)
    plt.scatter(actual, predicted)
    range = [actual.min(), actual.max()]
    plt.plot(range, range, 'black')
    plt.xlabel("Base Teste")
    plt.ylabel("Predito")
   

    plt.show()
    
scatterPlot(Y_teste_apartir19, previsaoPond_X)


# In[19]:


previsao = pd.DataFrame({'Data':t_teste_apartir19, 'Teste':Y_teste_apartir19, 'Predito':previsaoPond_X})
previsaolong = pd.melt(previsao, ['Data'] ,var_name='variavel', value_name='valores')


# In[20]:


fig  =  px . line (previsaolong,  x = "Data" ,  y = "valores" ,  color ='variavel',color_discrete_sequence=px.colors.qualitative.T10, template='plotly_white', title='Análise do valor predito com a base de teste', labels={'valores': 'vendas (R$)'}) 
fig .show ()


# In[110]:


plt.subplots(figsize=(17,5), sharex=True)

plt.plot(t_teste_apartir19,Y_teste_apartir19,label='Teste')
plt.plot(t_teste_apartir19,previsaoPond_X,label='Predito')
plt.legend()
plt.title('Previsão Vendas - Modelo com melhor desempenho')
plt.xlabel('Período Base Teste', fontsize = 13)
plt.ylabel('Vendas', fontsize = 13)
#plt.xlabel('Data', fontsize = 15) 
#plt.ylabel('Vendas (R$)', fontsize = 15) 
plt.xticks(rotation=45, fontsize = 13)

plt.xticks(t_teste_apartir19[::2], rotation=90)
plt.xticks([])
## Alterar espaçamento eixo x


# ### Regressão

# In[124]:


x_real4 = x_real3[x_real3.index < '2021-01-01']


# In[125]:


X = x_real4
y = Y_treino_apartir19
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est_pond = est.fit()
print(est_pond.summary())


# ## Validação Cruzada - K-folds
# 
# Uma das melhores técnicas para saber se o seu modelo generaliza bem, ou seja, como o modelo se comporta quando vai prever um dado que nunca viu.

# In[126]:


validaçao = dadosProntos[dadosProntos.index < '2021-04-01']


validaçao_X  = x_real3[x_real3.index < '2021-04-01']


# In[127]:


## Fazer Validação cruzada, é utilizada a base de dados inteira.
y = validaçao.Vendas
lm = linear_model.LinearRegression()
model = lm.fit(validaçao_X, y)
predictions = lm.predict(validaçao_X)


# In[128]:


import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import cross_val_score # Cross Validation Function.
from sklearn.model_selection import KFold # KFold Class.
from sklearn.linear_model import LinearRegression # Linear Regression class.
from sklearn.metrics import mean_squared_error

model  = LinearRegression()
kfold  = KFold(n_splits=10, shuffle=True) # shuffle=True, Shuffle (embaralhar) the data.
result = cross_val_score(model,validaçao_X, y, cv = kfold)

predictions = cross_val_predict(model,validaçao_X,y,cv=kfold)


print("K-Fold (R^2) Scores: {0}".format(result))
print("Mean R^2 for Cross-Validation K-Fold: {0}".format(result.mean()))
print("Verdadeiro Score que o modelo está generalizando  K-Fold: {0}".format(result.mean()))

print('Média: {:.2} | Desvio: {:.2}'.format(np.mean(result), np.std(result)))


# Agora temos o nosso R² para K iterações com dados de treinos e teste aleatórios. 

# ### Verificando a peformance de varios modelos
# 
# Criar uma função que veja a performance (R²) de vários modelos (Ex: Regressão) e escolha o melhor

# In[153]:


def ApplyesKFold(x_axis, y_axis):
  # Linear Models.
  from sklearn.linear_model import LinearRegression
  from sklearn.linear_model import ElasticNet
  from sklearn.linear_model import Ridge
  from sklearn.linear_model import Lasso

  # Cross-Validation models.
  from sklearn.model_selection import cross_val_score
  from sklearn.model_selection import KFold

  # KFold settings.
  kfold  = KFold(n_splits=10, shuffle=True) # shuffle=True, Shuffle (embaralhar) the data.

  # Axis
  x = x_axis
  y = y_axis

  # Models instances.
  linearRegression = LinearRegression()
  elasticNet       = ElasticNet()
  ridge            = Ridge()
  lasso            = Lasso()

  # Applyes KFold to models.
  linearRegression_result = cross_val_score(linearRegression, x, y, cv = kfold)
  elasticNet_result       = cross_val_score(elasticNet, x, y, cv = kfold)
  ridge_result            = cross_val_score(ridge, x, y, cv = kfold)
  lasso_result            = cross_val_score(lasso, x, y, cv = kfold)

  # Creates a dictionary to store Linear Models.
  dic_models = {
    "LinearRegression": linearRegression_result.mean(),
    "ElasticNet": elasticNet_result.mean(),
    "Ridge": ridge_result.mean(),
    "Lasso": lasso_result.mean()
  }
  # Select the best model.
  bestModel = max(dic_models, key=dic_models.get)

  print("Linear Regression Mean (R^2): {0}\nElastic Net Mean (R^2): {1}\nRidge Mean (R^2): {2}\nLasso Mean (R^2): {3}".format(linearRegression_result.mean(), elasticNet_result.mean(), ridge_result.mean(), lasso_result.mean()))
  print("The best model is: {0} with value: {1}".format(bestModel, dic_models[bestModel]))


if __name__ =='__main__':
  import pandas as pd

  dadosProntos
        
  #x_real3 = teste.drop(['dia', 'mês', 'ano', 'Dia_da_Semana', 'Dia_da_semana', 'semana_do_mês', 'Vendas', 'pond_mensal'], axis =1)
  validaçao_X
  #y = teste['Vendas']
  validaçao = dadosProntos[dadosProntos.index < '2021-04-01']
  y = validaçao['Vendas']
  ApplyesKFold(validaçao_X, y)


# ## 5. Embaralhando as observações

# In[130]:


UnidadeMatriz = dadosProntos.sample(frac=1)


# #### 5.1. Separando treino e teste

# In[131]:


Y_treino_data = UnidadeMatriz.Vendas[:722].index.values
Y_teste_data = UnidadeMatriz.Vendas[722:].index.values
Y_treino = UnidadeMatriz.Vendas[:722].values
Y_teste =  UnidadeMatriz.Vendas[722:].values

x = UnidadeMatriz.drop(['dia', 'mês', 'ano', 'Dia_da_semana', 'data', 'Vendas', 'mês_ano'], axis =1)

X_treino_data = x[:722].index.values
X_teste_data = x[722:].index.values
X_treino = x[:722].values
X_teste =  x[722:].values

## Checar se não há data repetida na base treino e na teste
#interc = pd.merge(treino, teste, on=["Data"], how="inner")


# In[132]:


model = LinearRegression().fit(X_treino, Y_treino)


# In[134]:


previsao = model.predict(X_teste)
residuo = Y_teste - previsao

print(model.score(X_treino, Y_treino)) 


# In[158]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

R2 = r2_score(Y_teste, previsao)
MAE = mean_absolute_error(Y_teste, previsao)
RMSE = np.sqrt(mean_squared_error(Y_teste, previsao))
print("MAE = {:0.2f}".format(MAE))
print("RMSE = {:0.2f}".format(RMSE))
print("R2= {:0.2f}".format(R2))

