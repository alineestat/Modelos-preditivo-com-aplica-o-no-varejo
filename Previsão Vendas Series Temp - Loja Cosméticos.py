#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install plotnine')


# In[2]:


get_ipython().system('pip install plotly')


# In[3]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import os,sys
from scipy import stats
import numpy as np

import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.stattools import adfuller

from scipy.stats import norm

from statsmodels.tsa.statespace.sarimax import SARIMAX
import statistics 

import statsmodels.api   as sm
import seaborn           as sb

from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tools.eval_measures import rmse
get_ipython().run_line_magic('matplotlib', 'inline')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

#!pip install plotnine
#!pip install plotly
import plotnine
from plotnine import *

import plotly.express as px
import plotly.graph_objects as go


from scipy.stats import boxcox
from scipy.special import inv_boxcox
from pmdarima.arima import ARIMA

from pmdarima.arima import auto_arima

pd.options.display.max_rows = 999
pd.set_option("display.float_format", lambda x: "%.2f" %x)

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[4]:


get_ipython().system(' pip install pmdarima')


# ## Previsão Vendas Séries Temporais e Suavização exponencial 

# ## 1. Banco de Dados

# In[5]:


dados = pd.read_excel("C:/Users/aline.nunes/Documents/ProjetoSupervisionado/Modelagem Matriz-20221118T223710Z-001/Modelagem Matriz/Unidade Matriz/DadosLojaCosmeticos.xlsx")

colunas = ['dia', 'mês', 'ano', 'Dia_da_semana', 'Num_Semana_mes', 'feriados', 'pagamento', 'Vale']
for i in range(len(colunas)):
    dados[colunas[i]] = dados[colunas[i]].astype(int)
dados = dados[dados['Data'] < '2021-04-01']
dados


# In[ ]:


### Inserindo Informações Públicas


# In[6]:


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


# ## 2. Análise Exploratória
# 
# * #### 2.1 Gráfico da Séria
# * #### 2.2 Gráfico Vendas Anuais e Mensais
# 
#         Gráfico de linhas das vendas anuais. Boxplot das vendas por ano e mês
# 
# * #### 2.3 Correlograma
# * #### 2.4 Análise de Outliers
# * #### 2.5 Decomposição da série em Tendência e Sazonalidade
# * #### 2.6 Teste de Estacionaridade
# * #### 2.7 Análise FAC e FACP da série
# * #### 2.8 Transformação Log
# 

# * ### 2.1 Gráfico da Série

# In[7]:


fig = px.line(dados, x="Data", y="Vendas", template='plotly_white',labels={'Vendas': 'Vendas (R$)'},color_discrete_sequence=px.colors.qualitative.T10)
fig.update_layout(title={
    'text' : 'Vendas Diárias do ano de 2018 até março 2021'})


# * ### 2.2 Gráfico das Vendas Anuais e mensais

# In[8]:


dados['Mês'] = dados['mês'].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"])

import plotly.graph_objects as go

fig_vendas_anual = px.line(dados.groupby(by=["mês","Mês", "ano"])["Vendas"].sum().reset_index(), x="Mês", y="Vendas", color="ano", labels={'Vendas': 'Vendas (R$)'})
fig_vendas_anual.update_layout(title={
    'text' : 'Comparativo das vendas por mês nos anos de estudo'}, xaxis_title='Mês', template='plotly_white')
#fig.update_xaxes(dtick="M1")
fig_vendas_anual



# In[9]:


fig1 = px.box(dados[dados['ano'] < 2021], x="Mês", y="Vendas", labels={'Vendas': 'Vendas (R$)'},color_discrete_sequence=px.colors.qualitative.T10)
fig1.update_layout(title={
    'text' : 'Diagrama das vendas por mês, dados 2018 até 2020'}, xaxis_title='Mês', template='plotly_white')

fig2 = px.box(dados[dados['ano'] < 2021], x="ano", y="Vendas", labels={'Vendas': 'Vendas (R$)'},color_discrete_sequence=px.colors.qualitative.T10)
fig2.update_layout(title={
    'text' : 'Diagrama das vendas por ano'}, xaxis_title='Ano', template='plotly_white')
#fig.update_xaxes(dtick="M1")
fig1.show(), 
fig2.show()


# In[10]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='ano', y='Vendas', data=dados, ax=ax1)
sns.boxplot(x='Mês', y='Vendas', data=dados, ax=ax2)
plt.show()


# * ### 2.4 Análise de Outliers

# In[11]:


outlier = px.box(dados, y="Vendas",color_discrete_sequence=px.colors.qualitative.T10, template='plotly_white')
outlier.update_layout(title={
    'text' : 'Outliers de acorco dia da semana'})
print(outlier.show())


# #### Identificando Outliers com Z-Score
# 
# O Z-Score nos da a idéai do quanto um determidado ponto está afastado  da média dos dados.
# Mede quantos desvios padrões abaixo ou acima da média dos dados estão. È dado por:
# 
# $$z= \frac{x - \mu}{\sigma}$$
# 
# Quanto mais longo o Z-score de uma observação está de zero, mais incomum ele é. Um valor de corte padrão para encontrar outliers são escores Z de +/-3 ou mais longe de zero. 

# In[12]:


outliers=[]


def find_outliers(data_set):
    
    corte_dp = 3
    media = np.mean(data_set)
    dp = np.std(data_set)
    
    for dado in data_set:
        
        z_score = (dado-media)/dp
        if np.abs(z_score) >= corte_dp:
            outliers.append(dado)
            
            
    return outliers
    

outliers = find_outliers(dados.Vendas)

outliers = pd.DataFrame(outliers)
outliers = outliers.rename(columns = {0:'Vendas'})


# In[13]:


dados_outliers = pd.merge(dados, outliers, how = "inner", on= ["Vendas"])
dados_outliers


# #### Adição Outliers conjunto de dados

# In[14]:


conditions = [
    (dados['Data'] == '2019-11-29'),
    (dados['Data'] == '2020-11-27')]
choices = [1, 1]

dados['BlackFriday'] = np.select(conditions, choices, default= 0)

conditions2 = [
    (dados['Data'] == '2018-03-31'),
    (dados['Data'] == '2019-04-20'),
    (dados['Data'] == '2020-04-11')]
choices2 = [1, 1, 1]

dados['SabadoAleluia'] = np.select(conditions2, choices2, default= 0)

#filtro =dados_Frei_ajustado[(dados_Frei_ajustado['Data'] == '2019-11-29')]


# * ### 2.5 Decomposição da série em Tendência e Sazonalidade

# In[15]:


dados["ano"] = dados["ano"].astype(int)
decomposicao = dados[dados["ano"] > 2018]
decomposicao = decomposicao.groupby(by=["Data"])["Vendas"].sum()
decomposicao =decomposicao.reset_index()
decomposicao.set_index('Data', inplace=True)


decomposicao_treino = dados[(dados["ano"] > 2018) & (dados["ano"] <= 2020)]
decomposicao_treino = decomposicao_treino.groupby(by=["Data"])["Vendas"].sum()
decomposicao_treino = decomposicao_treino.reset_index()
decomposicao_treino.set_index('Data', inplace=True)


decomposicao_teste = dados[dados["ano"] > 2020]
decomposicao_teste = decomposicao_teste.groupby(by=["Data"])["Vendas"].sum()
decomposicao_teste = decomposicao_teste.reset_index()
decomposicao_teste.set_index('Data', inplace=True)


# In[16]:


from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = (11, 13)

# Multiplicative Decomposition 
result_mul = seasonal_decompose(decomposicao, model='multiplicative', extrapolate_trend='freq')
result_mul.plot().suptitle('Multiplicative Decompose');

# Additive Decomposition
result_add = seasonal_decompose(decomposicao, model='additive', extrapolate_trend='freq')
result_add.plot().suptitle('Additive Decompose');


# In[17]:


def check_mean_std(ts):
    #Rolling statistics
    rolmean = ts.rolling(12).mean()
    rolstd = ts.rolling(12).std()
    plt.figure(figsize=(22,10))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()
    
check_mean_std(dados.Vendas)


# * ### 2.6 Teste Estacionaridade

# A série é considerada estacionária se:
# 
# * E(X) = const. para todo t
# * Var(X) = const. para todo t
# * Cov($Y_{t}, Y_{t+h})$ é função somente de h
# 
# Será utilizado teste de Dickey-Fuller para ajudar a entender se a série é estacionária ou não
# 
# $H_{0}$ = série Não é estacionária
# 
# $H_{1}$ = É estacionária
# 
# Se a estatística do teste > -3,12 não rejeito a hipotese nula
# <p>Se a estatística do teste < -3,12 rejeito a hipótese nula
# 
# Caso p-valor seja menor que $\alpha$ = 0.05 rejeito $H_{0}$
# 
# Caso p-valor seja maior que $\alpha$ = 0.05 aceito $H_{0}$, ou seja, a série Não é estacionária

# In[18]:


## Considerando a Base de dados apartir do ano 2019

adfuller1_3 = adfuller(decomposicao, autolag='AIC')
print(f'ADF Statistic: {adfuller1_3[0]}')
print(f'p-value: {adfuller1_3[1]}')
for key, value in adfuller1_3[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
#print(adfuller1_3)

# KPSS Test
result = kpss(decomposicao, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


# Fazendo o teste de Dickey-Fuller foi obtido p-valor = 0, portanto ao nível de significância de 5% a série é estacionária.

# * ### 2.7 Análise FAC e FACP 

# #### FAC e FACP da série SEM diferenciação

# In[19]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
plot_acf(decomposicao, lags=120, ax=ax1) ## Especificando qtd de lags

plot_pacf(decomposicao, lags= 120, ax=ax2) 
plt.show()


# #### Fac e FACP da série diferenciada na parte sazonal
# 
# Sazonalidade semanal

# In[20]:


dif = decomposicao.diff(7)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 7))
plot_acf(dif.dropna(), lags=120, ax=ax1)  ## Especificando qtd de lags
plot_pacf(dif.dropna(), lags= 120, ax=ax2) 

plt.show()


# In[21]:


dif.plot(figsize = (15,6))


# In[22]:


# Vamos agora realizar os testes novamente de ruído branco

#Teste Ljung-Box

import statsmodels.api as sm
sm.stats.acorr_ljungbox(decomposicao, lags=[7], return_df=True)
#p_valor < nível_significancia  => rejeitar a hipotese nula => série não é ruído branco

# Como o valor p-valor foi muito baixo, NÃO HÁ RUIDO BRANCO


# ## 3. Separando em treino e teste

# In[23]:


dados = dados[dados['Data'] > '2018-12-31']


# In[24]:


tam_treino = int(len(dados.Vendas)*2/3) 
tam_teste = int(len(dados.Vendas)*1/3) 
tam_teste 


# In[25]:


tam_treino


# In[32]:


## Foram utilizadas modelo SARIMA

dados.set_index('Data', inplace=True)

t_treino_apartir19 = dados.Vendas[dados.Vendas.index <'2020-07-01'].index.values
t_teste_apartir19 = dados.Vendas[dados.Vendas.index >= '2020-07-01'].index.values
X_treino_apartir19 = dados.Vendas[dados.Vendas.index <'2020-07-01'].values
X_teste_apartir19 = dados.Vendas[dados.Vendas.index >= '2020-07-01'].values


exog = dados.drop(['dia', 'mês', 'ano', 'DiasNaoUteis','Dia_Úteis','Vendas', 'mês_ano', 'Dia_da_Semana', 'Mês', 'Dia_da_semana', 'Num_Semana_mes'],axis=1)
treino_exog = exog[exog.index < '2020-07-01']
teste_exog = exog[exog.index >= '2020-07-01']


# In[33]:


t_treino_apartir19


# ### Função Correlação Cruzada

# In[ ]:


x=dados.Vendas
y=dados.feriados

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(15,4), sharex=True)
plt.title('Função Correlação Cruzada ( Vendas X feriados)')
ax1.xcorr(x, y, usevlines=True, maxlags=50, normed=True, lw=2)
ax1.grid(True)

ax2.acorr(x, usevlines=True, normed=True, maxlags=50, lw=2)
ax2.grid(True)

plt.show()


# ## 4 Modelagem
# 
# * #### 4.1 Modelos Exponenciais
# * #### 4.2 Modelo SARIMA
#   #### 4.2 1. Modelo SARIMA SIMPLES - Critério BIC
#           
#           Modelo candidato: SARIMA (4,0,3)(0,1,1)7.
#           Não passou nos critérios de independência e normalidade.
#           Foi obtido RMSE = 82902.74
#           
#   #### 4.2 2. Modelo SARIMA Adição Variáveis exogenas - Critério BIC
#   
#           Modelo candidato: SARIMA (2,0,3)(0,1,1)7
#           Não passou nos critérios de independência e normalidade.
#           Foi obtido RMSE = 83825.33
#          
#   #### 4.2 3. Modelo AutoArima - Critério AIC
#           
#           Modelo candidato: SARIMA (1,0,3)(0,1,1)7
#           Não passou nos critérios de independência e normalidade.
#           Foi obtido RMSE = 44641.154
#           R2 = 0,50
#           
#   #### 4.2 4. Escolha modelo pelo AutoArima com adição variáveis exogenas
#   
#           Modelo candidato: SARIMA (3,0,1)(0,1,1)7
#           Não passou nos critérios de independência e normalidade.
#           Foi obtido RMSE = 99792.14
#          

# * ### 4.1 Modelos Exponenciais

# #### 4.* Método exponencial para tendência e sazonalidade - lag = 7

# In[ ]:


from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[ ]:


# Comparação de modelos Holt Winters com diferenciação de sazonalidade aditivo e multiplicativo, com Damping e sem Damping
fit1 = ExponentialSmoothing(decomposicao, seasonal_periods=7, trend='add', seasonal='add', use_boxcox=True, initialization_method="estimated").fit()
fit2 = ExponentialSmoothing(decomposicao, seasonal_periods=7, trend='add', seasonal='mul', use_boxcox=True, initialization_method="estimated").fit()
fit3 = ExponentialSmoothing(decomposicao, seasonal_periods=7, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
fit4 = ExponentialSmoothing(decomposicao, seasonal_periods=7, trend='add', seasonal='mul', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
results["Additive"]       = [fit1.params[p] for p in params] + [fit1.sse]
results["Multiplicative"] = [fit2.params[p] for p in params] + [fit2.sse]
results["Additive Dam"]   = [fit3.params[p] for p in params] + [fit3.sse]
results["Multiplica Dam"] = [fit4.params[p] for p in params] + [fit4.sse]

ax = decomposicao.plot(figsize=(15,6), marker='o', color='black', title="Forecasts from Holt-Winters' multiplicative method" )
ax.set_ylabel("Vendas")
ax.set_xlabel("Ano")
fit1.fittedvalues.plot(ax=ax, style='--', color='red')
fit2.fittedvalues.plot(ax=ax, style='--', color='green')

fit1.forecast(59).rename('Holt-Winters (add-add-seasonal)').plot(ax=ax, style='--', marker='o', color='red', legend=True)
fit2.forecast(59).rename('Holt-Winters (add-mul-seasonal)').plot(ax=ax, style='--', marker='o', color='green', legend=True)

# The plot shows the results and forecast for fit1 and fit2. The table allows us to compare the results and parameterizations.
plt.show()
print("Vendas por Ano - métodos Holt-Winters sazonalidade aditiva vs multiplicativa.")

results


# <p> Onde $\alpha$ é o parâmetro de suavização para o nível.
# <p> Onde $\beta$ é o prâmetro de suavização para tendência.
# <p> Onde $\gamma$ é o prâmetro de suavização para sazonalidade.

# In[ ]:


fit1 = ExponentialSmoothing(decomposicao, seasonal_periods=7, trend='add', seasonal='add', use_boxcox=True, initialization_method="estimated").fit()
fit2 = ExponentialSmoothing(decomposicao, seasonal_periods=7, trend='add', seasonal='mul', use_boxcox=True, initialization_method="estimated").fit()
fit3 = ExponentialSmoothing(decomposicao, seasonal_periods=7, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
fit4 = ExponentialSmoothing(decomposicao, seasonal_periods=7, trend='add', seasonal='mul', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()

fcast1_htw = fit1.forecast(12).rename("Holt Winters Multiplicativo")
print(fit1.summary())  

fcast2_htw = fit2.forecast(12).rename("Holt Winters Multiplicativo")
print(fit2.summary())  


# #### Acurácia Modelos Alisamento Exponencial

# In[ ]:


y_true = decomposicao
y_pred_fit1 = fit1.fittedvalues
r2_fit1   = r2_score(y_true, y_pred_fit1)
mse_fit1  = mean_squared_error(y_true, y_pred_fit1)
rmse_fit1 = sqrt(mean_squared_error(y_true, y_pred_fit1))
mae_fit1  = mean_absolute_error(y_true, y_pred_fit1)
mape_fit1 = mean_absolute_percentage_error(y_true, y_pred_fit1)

print('fit1- lag7')
print('R2:   %.3f' % r2_fit1)
print('MSE:  %.3f' % mse_fit1)
print('RMSE: %.3f' % rmse_fit1)
print('MAE:  %.3f' % mae_fit1)
print('MAPE: %.3f' % mape_fit1)
print("*****************************")

y_true = decomposicao
y_pred_fit2 = fit2.fittedvalues
r2_fit2   = r2_score(y_true, y_pred_fit2)
mse_fit2  = mean_squared_error(y_true, y_pred_fit2)
rmse_fit2 = sqrt(mean_squared_error(y_true, y_pred_fit2))
mae_fit2 = mean_absolute_error(y_true, y_pred_fit2)
mape_fit2 = mean_absolute_percentage_error(y_true, y_pred_fit2)

print('fit2- lag7')
print('R2:   %.3f' % r2_fit2)
print('MSE:  %.3f' % mse_fit2)
print('RMSE: %.3f' % rmse_fit2)
print('MAE:  %.3f' % mae_fit2)
print('MAPE: %.3f' % mape_fit2)
print("*****************************")

y_true = decomposicao
y_pred_fit3= fit3.fittedvalues
r2_fit3   = r2_score(y_true, y_pred_fit3)
mse_fit3  = mean_squared_error(y_true, y_pred_fit3)
rmse_fit3 = sqrt(mean_squared_error(y_true, y_pred_fit3))
mae_fit3  = mean_absolute_error(y_true, y_pred_fit3)
mape_fit3 = mean_absolute_percentage_error(y_true, y_pred_fit3)

print('fit3- lag7')
print('R2:   %.3f' % r2_fit3)
print('MSE:  %.3f' % mse_fit3)
print('RMSE: %.3f' % rmse_fit3)
print('MAE:  %.3f' % mae_fit3)
print('MAPE: %.3f' % mape_fit3)
print("*****************************")

y_true = decomposicao
y_pred_fit4 = fit4.fittedvalues
r2_fit4   = r2_score(y_true, y_pred_fit4)
mse_fit4  = mean_squared_error(y_true, y_pred_fit4)
rmse_fit4 = sqrt(mean_squared_error(y_true, y_pred_fit4))
mae_fit4 = mean_absolute_error(y_true, y_pred_fit4)
mape_fit4 = mean_absolute_percentage_error(y_true, y_pred_fit4)

print('fit4- lag7')
print('R2:   %.3f' % r2_fit4)
print('MSE:  %.3f' % mse_fit4)
print('RMSE: %.3f' % rmse_fit4)
print('MAE:  %.3f' % mae_fit4)
print('MAPE: %.3f' % mape_fit4)
print("*****************************")


# In[ ]:


#Teste Ljung-Box
sm.stats.acorr_ljungbox(decomposicao, lags=[7], return_df=True)

print(sm.stats.acorr_ljungbox(decomposicao, lags=[7], return_df=True))
print('Conclusão: NÃO É RUIDO BRANCO')


# #### Alisamento exponencial, período igual a 14

# In[ ]:


#Comparação de modelos Holt Winters com diferenciação de sazonalidade aditivo e multiplicativo, com Damping e sem Damping
fit1_lag14 = ExponentialSmoothing(decomposicao, seasonal_periods=14, trend='add', seasonal='add', use_boxcox=True, initialization_method="estimated").fit()
fit2_lag14 = ExponentialSmoothing(decomposicao, seasonal_periods=14, trend='add', seasonal='mul', use_boxcox=True, initialization_method="estimated").fit()
fit3_lag14 = ExponentialSmoothing(decomposicao, seasonal_periods=14, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
fit4_lag14 = ExponentialSmoothing(decomposicao, seasonal_periods=14, trend='add', seasonal='mul', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
results["Additive"]       = [fit1_lag14.params[p] for p in params] + [fit1.sse]
results["Multiplicative"] = [fit2_lag14.params[p] for p in params] + [fit2.sse]
results["Additive Dam"]   = [fit3_lag14.params[p] for p in params] + [fit3.sse]
results["Multiplica Dam"] = [fit4_lag14.params[p] for p in params] + [fit4.sse]

ax = decomposicao.plot(figsize=(15,6), marker='o', color='black', title="Forecasts from Holt-Winters' multiplicative method" )
ax.set_ylabel("Vendas")
ax.set_xlabel("Ano")
fit1_lag14.fittedvalues.plot(ax=ax, style='--', color='red')
fit2_lag14.fittedvalues.plot(ax=ax, style='--', color='green')

fit1_lag14.forecast(12).rename('Holt-Winters (add-add-seasonal)').plot(ax=ax, style='--', marker='o', color='red', legend=True)
fit2_lag14.forecast(12).rename('Holt-Winters (add-mul-seasonal)').plot(ax=ax, style='--', marker='o', color='green', legend=True)

# The plot shows the results and forecast for fit1 and fit2. The table allows us to compare the results and parameterizations.
plt.show()
print("Vendas por Ano - métodos Holt-Winters sazonalidade aditiva vs multiplicativa.")

results


# In[ ]:


y_true = decomposicao
y_pred_fit1_lag14 = fit1_lag14.fittedvalues
r2_fit1_lag14   = r2_score(y_true, y_pred_fit1_lag14)
mse_fit1_lag14  = mean_squared_error(y_true, y_pred_fit1_lag14)
rmse_fit1_lag14 = sqrt(mean_squared_error(y_true, y_pred_fit1_lag14))
mae_fit1_lag14  = mean_absolute_error(y_true, y_pred_fit1_lag14)
mape_fit1_lag14 = mean_absolute_percentage_error(y_true, y_pred_fit1_lag14)

print('fit1- lag14')
print('R2:   %.3f' % r2_fit1_lag14)
print('MSE:  %.3f' % mse_fit1_lag14)
print('RMSE: %.3f' % rmse_fit1_lag14)
print('MAE:  %.3f' % mae_fit1_lag14)
print('MAPE: %.3f' % mape_fit1_lag14)
print("*****************************")

y_true = decomposicao
y_pred_fit2_lag14 = fit2_lag14.fittedvalues
r2_fit2_lag14   = r2_score(y_true, y_pred_fit2_lag14)
mse_fit2_lag14  = mean_squared_error(y_true, y_pred_fit2_lag14)
rmse_fit2_lag14 = sqrt(mean_squared_error(y_true, y_pred_fit2_lag14))
mae_fit2_lag14  = mean_absolute_error(y_true, y_pred_fit2_lag14)
mape_fit2_lag14 = mean_absolute_percentage_error(y_true, y_pred_fit2_lag14)

print('fit2- lag14')
print('R2:   %.3f' % r2_fit2_lag14)
print('MSE:  %.3f' % mse_fit2_lag14)
print('RMSE: %.3f' % rmse_fit2_lag14)
print('MAE:  %.3f' % mae_fit2_lag14)
print('MAPE: %.3f' % mape_fit2_lag14)
print("*****************************")

y_true = decomposicao
y_pred_fit3_lag14 = fit3_lag14.fittedvalues
r2_fit3_lag14   = r2_score(y_true, y_pred_fit3_lag14)
mse_fit3_lag14  = mean_squared_error(y_true, y_pred_fit3_lag14)
rmse_fit3_lag14 = sqrt(mean_squared_error(y_true, y_pred_fit3_lag14))
mae_fit3_lag14  = mean_absolute_error(y_true, y_pred_fit3_lag14)
mape_fit3_lag14 = mean_absolute_percentage_error(y_true, y_pred_fit3_lag14)

print('fit3- lag14')
print('R2:   %.3f' % r2_fit3_lag14)
print('MSE:  %.3f' % mse_fit3_lag14)
print('RMSE: %.3f' % rmse_fit3_lag14)
print('MAE:  %.3f' % mae_fit3_lag14)
print('MAPE: %.3f' % mape_fit3_lag14)
print("*****************************")

y_true = decomposicao
y_pred_fit4_lag14 = fit4_lag14.fittedvalues
r2_fit4_lag14   = r2_score(y_true, y_pred_fit4_lag14)
mse_fit4_lag14  = mean_squared_error(y_true, y_pred_fit4_lag14)
rmse_fit4_lag14 = sqrt(mean_squared_error(y_true, y_pred_fit4_lag14))
mae_fit4_lag14  = mean_absolute_error(y_true, y_pred_fit4_lag14)
mape_fit4_lag14 = mean_absolute_percentage_error(y_true, y_pred_fit4_lag14)

print('fit4- lag14')
print('R2:   %.3f' % r2_fit4_lag14)
print('MSE:  %.3f' % mse_fit4_lag14)
print('RMSE: %.3f' % rmse_fit4_lag14)
print('MAE:  %.3f' % mae_fit4_lag14)
print('MAPE: %.3f' % mape_fit4_lag14)


# #### Melhor Modelo Exponencial
# 
# Comparando o modelo exponencial com lag 7 e lag 14 percebe-se que o modelo exponencial com lag 7 apresenta uma performace melhor. O modelo com melhor ajuste foi o fit3. 
# 
# Modelo exponencial Aditivo, com Damping
# 
# fit3 = ExponentialSmoothing(decomposicao, seasonal_periods=7, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()

# * ### 4.2 Modelo SARIMA

#    #### 4.2 1. Modelo SARIMA SIMPLES - Critério BIC e AIC

#    #### 4.2 3. Escolha modelo pelo AutoArima, usando critério AIC

# In[34]:


apartir19 = dados.Vendas[dados.Vendas.index <'2020-07-01'].reset_index()
apartir19.set_index('Data', inplace=True)

## Auto_Arima M
arima_model = auto_arima(apartir19,
                         start_p=0, d=0, start_q=0, 
                         max_p=8, max_q=8, 
                         start_P=0, D=1, start_Q=0, 
                         max_P=8, max_Q=8, 
                         m=7, seasonal=True, 
                         trace=True, 
                         error_action='ignore',
                         supress_warnings=True, stepwise = True , information_criterion='aic')

arima_model.summary()


# In[36]:


yhat = arima_model.predict_in_sample()
apartir19['yhat'] = yhat

plt.subplots(figsize=(18,7), sharex=True)
plt.plot(apartir19.Vendas,label='Treino')
plt.plot(apartir19.yhat, label='Predito')
plt.xticks(rotation=45)
plt.legend()




# 3. AVALIAÇÃO base treino
# Accuracy metrics

from sklearn.metrics import r2_score

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    r2 = r2_score(forecast, actual)
    return({'Base Treino MAPE':mape, 'Base Treino ME':me, 'Base Treino MAE': mae, 
            'Base Treino MPE': mpe, 'Base Treino RMSE':rmse, 'corr':corr, 'Base Treino R2':r2})

forecast_accuracy(apartir19.yhat,apartir19.Vendas)


# In[37]:


################################
# 4. VERIFICAÇÃO
# Avalia a qualidade do ajuste 
# Aplicando teste do ruido branco do resíduo
sm.stats.acorr_ljungbox(arima_model.resid(), lags=[10], return_df=True)


# In[38]:


arima_modell = SARIMAX(X_treino_apartir19, order=(3,0,0), seasonal_order = (0,1,2,7)).fit()


#fig=arima_modell.plot_diagnostics()

arima_modell.plot_diagnostics(figsize=(16,8))
plt.show()

## Adicionado pela Aline

import statsmodels.api as sm
from pandas import DataFrame

residuo_modelo = DataFrame(arima_modell.resid)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))
## Plot da FAC dos resíduos
plot_acf(residuo_modelo,lags=25, ax=ax1)


## Plot da FACp dos resíduos
plot_pacf(residuo_modelo, lags=25, ax=ax2)
plt.show()

sm.stats.acorr_ljungbox(arima_modell.resid, lags=[14], return_df=True)
#Esta função retorna uma estatística de teste e um valor p correspondente. Se o valor p for inferior a um nível de significância (por exemplo, α = 0,05), a evidências para rejeitar a hipótese nula e concluir que os resíduos não são distribuídos independentemente.

#shapiro_test = stats.shapiro(residuo_modelo)
#print("", shapiro_test )


# #### Verificando desempenho modelo SARIMA, com parâmetros definidos no auto-arima base treino

# In[39]:


################################
# 5. FORECAST
apartir19_teste = dados.Vendas[dados.Vendas.index >= '2020-07-01'].reset_index()
apartir19_teste.set_index('Data', inplace=True)


n_periods = 274 #tamanho da base de teste
fitted, confint = arima_model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(apartir19_teste.index[0], periods = n_periods, freq='D')


# In[40]:


# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)


# In[65]:


# Plot real x predito - Base teste

plt.subplots(figsize=(20,7), sharex=True)
plt.plot(apartir19_teste, label='Base de Teste')
plt.plot(fitted_series,  label='Valores Preditos')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.00)
plt.xlabel('Data', fontsize = 15) 
plt.ylabel('Vendas (R$)', fontsize = 15) 
plt.xticks(rotation=45, fontsize = 13)

plt.legend(fontsize = 14)
plt.title("SARIMA", fontsize = 15)
plt.show()


# In[243]:


# Plot real x predito - Base train + teste

plt.subplots(figsize=(17,4), sharex=True)
plt.plot(apartir19, color='blue')
plt.plot(apartir19_teste, color='darkblue')
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)
plt.xticks(rotation=45)
plt.title("SARIMA - Base Treino e Teste")
plt.show()


# BASE TESTE

# In[244]:


y_true = apartir19_teste.Vendas
y_pred = fitted

r2   = r2_score(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print('Base Teste R2:   %.3f' % r2)
print('Base Teste MSE:  %.3f' % mse)
print('Base Teste RMSE: %.3f' % rmse)
print('Base Teste MAE:  %.3f' % mae)
print('Base Teste MAPE: %.3f' % mape)


#  #### 4.2 3. Escolha modelo pelo AutoArima, usando critério BIC

# In[114]:


apartir19 = apartir19[['Vendas']]
arima_model_BIC = auto_arima(apartir19, start_p =0, d=0, start_q=0,
                            max_p=8, max_d=0, max_q=8, start_P=0,
                            D=1, start_Q=0, m=7, max_P=8, max_D=1,
                            max_Q=8, seasonal=True, information_criterion='bic',
                            error_action='warn', supress_warnings=True, trace= True, stepwise=False,
                            randow_state=20, n_fits=15)

arima_model_BIC.summary()


# ### ADIÇÃO VARIÁVEIS EXÓGENAS

#    #### 4.2 4. Escolha modelo pelo AutoArima com adição variáveis exogenas
# 
# Variáveis exógenas consideradas:
# 
# * Dia_da_semana
# * feriados
# * pagamento
# * semana_do_mês
# * Outlier_DecTerceiro
# * Outlier_InicPandemia

# In[259]:


treino_exog


# In[260]:


modelo_checagem = auto_arima(t_treino_apartir19,exogenous=treino_exog , start_p =0, d=0, start_q=0,
                            max_p=8, max_d=0, max_q=8, start_P=0,
                            D=1, start_Q=0, m=7, max_P=8, max_D=1,
                            max_Q=8, seasonal=True, error_action='warn', trace= True, stepwise=False, information_criterion='bic')


# In[262]:


apartir19 = apartir19[['Vendas']]
## Rodando AutoArima com adição das variáveis exogenas foi obtido que o melhor modelo é: SARIMA (1,0,2)(0,1,2)
modelo_sarima_exog = SARIMAX(endog = apartir19, exog= treino_exog, order=(0,0,0), seasonal_order = (0,1,0,7)).fit()
                
print(modelo_sarima_exog.summary())

modelo_sarima_exog.plot_diagnostics(figsize=(16,8))
plt.show()


import statsmodels.api as sm
from pandas import DataFrame

residuo_modelo_sarima_exog = DataFrame(modelo_sarima_exog.resid)
print(modelo_sarima_exog)

## FAC e FACP residual
plt.figure()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
plot_acf(residuo_modelo_sarima_exog,lags=20, ax = ax1)

plot_pacf(residuo_modelo_sarima_exog, lags=20, ax = ax2)
plt.show()

shapiro_test = stats.shapiro(residuo_modelo_sarima_exog)
print("", shapiro_test )

print("Teste ljungbox")
sm.stats.acorr_ljungbox(modelo_sarima_exog.resid, lags=[14], return_df=True)



# Métrica Acurácia Base treino

# In[187]:


y_hat_train = modelo_sarima_exog.forecast(steps=treino_exog.shape[0],exog=treino_exog)
len(y_hat_train)


y_true = apartir19
y_pred = y_hat_train
r2   = r2_score(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print('R2:   %.3f' % r2)
print('MSE:  %.3f' % mse)
print('RMSE: %.3f' % rmse)
print('MAE:  %.3f' % mae)
print('MAPE: %.3f' % mape)


# Métrica Acurácia Base Teste

# In[188]:


y_hat_test = modelo_sarima_exog.forecast(steps=teste_exog.shape[0],exog=teste_exog)
len(y_hat_test)

y_true = apartir19_teste
y_pred = y_hat_test
r2   = r2_score(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print('R2:   %.3f' % r2)
print('MSE:  %.3f' % mse)
print('RMSE: %.3f' % rmse)
print('MAE:  %.3f' % mae)
print('MAPE: %.3f' % mape)

plt.plot(y_true, color='red',label='Teste')
plt.plot(y_pred, color='black', label='predito')
plt.title("Previsão Base Teste modelo Sarima (1,0,2)(0,1,2)7")

plt.legend()

