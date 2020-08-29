# Importar paquetes requeridos
import requests
import pandas as pd
from datetime import date
import time
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# model = ARIMA(test['casos_diff'], order=(5,1,0))
# %%

def extract_data(api_url, limit):
    # Funcion para la extraccion de los datos via SODA API ODATA
    data_len = 50000
    offset = 0
    appended_data = []
    while data_len >= limit:
        time.sleep(1)
        params = {'$limit': limit, '$offset': offset}
        response = requests.get(api_url, params=params)
        data = response.json()
        data_len = len(data)
        if 'error' not in data or len(data) > 0:
            df = pd.DataFrame.from_dict(data, orient='columns')
            df['extracted_at_utc'] = pd.to_datetime('now',utc=True)
            appended_data.append(df)
            offset = offset + limit
    return pd.concat(appended_data, ignore_index=True, sort=False)


def data_transform(df, start_date):
    # Funcion transformacion de los datos (Data quality)
    df_in = df
    if 'atenci_n' in df.columns:
        df_in['atenci_n'] = df['atenci_n'].str.strip()
        df_in['atenci_n'] = df['atenci_n'].str.lower()
    delta = date.today() - start_date
    df_dates = pd.DataFrame(pd.date_range(start_date, periods=delta.days, freq='D'), columns=['fecha'])
    df_in['fecha_de_notificaci_n'] = pd.to_datetime(df_in['fecha_de_notificaci_n'])
    df_in['fecha_de_muerte'] = pd.to_datetime(df_in['fecha_de_muerte'])
    df_in['fecha_recuperado'] = pd.to_datetime(df_in['fecha_recuperado'])
    return df_in, df_dates

def data_agg_col(df):
    # Funcion para realizar agregaciones
    df['c_recuperado'] = df['atenci_n'].apply(lambda x : 1 if x == 'recuperado' else 0)
    df['c_fallecido'] = df['atenci_n'].apply(lambda x: 1 if x == 'fallecido' else 0)
    df['c_caso'] = 1
    df_final = df.groupby(['fecha_de_notificaci_n'], as_index=False).agg({'id_de_caso': 'count', 'c_recuperado': 'sum', 'c_fallecido': 'sum'})
    df_casos = df.groupby(['fecha_de_notificaci_n'], as_index=False)['c_caso'].sum()
    df_casos = df_casos.rename(columns={'fecha_de_notificaci_n': 'fecha'})
    df_muertes = df.groupby(['fecha_de_muerte'], as_index=False)['c_fallecido'].sum()
    df_muertes = df_muertes.rename(columns={'fecha_de_muerte':'fecha'})
    df_recuperados = df.groupby(['fecha_recuperado'], as_index=False)['c_recuperado'].sum()
    df_recuperados = df_recuperados.rename(columns={'fecha_recuperado': 'fecha'})
    df_final = df_final.rename(columns={'id_de_caso':'total_casos','c_recuperado':'total_recuperados','c_fallecido':'total_fallecidos'})
    return df_casos, df_muertes, df_recuperados

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
        # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
#def data_agg_main_cities(df, main_cities)

'''from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, squeeze=True)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))'''

# %%
def main():
    # %%
    api_url = "https://www.datos.gov.co/resource/gt2j-8ykr.json"
    df_casos = extract_data(api_url, 50000)
    df_casos, df_dates = data_transform(df_casos,date(2020,3,1))
    df_casos_p,df_muertes,df_recuperados = data_agg_col(df_casos)
    df_full = pd.merge(df_dates, df_casos_p, how='left', on=['fecha'])
    df_full = pd.merge(df_full, df_muertes, how='left', on=['fecha'])
    df_full = pd.merge(df_full, df_recuperados, how='left', on=['fecha'])
    # df_full = df_full.fillna(0)
    df_full['casos_lag'] = df_full['c_caso'].shift(1)
    df_full['fallecidos_lag'] = df_full['c_fallecido'].shift(1)
    df_full['recuperados_lag'] = df_full['c_recuperado'].shift(1)
    df_full = df_full.replace({0: np.nan})
    df_full['casos_diff'] = (df_full['casos_lag'] - df_full['c_caso'])/df_full['c_caso']
    df_full['fallecidos_diff'] = (df_full['fallecidos_lag'] - df_full['c_fallecido']) / df_full['c_fallecido']
    df_full['recuperados_diff'] = (df_full['recuperados_lag'] - df_full['c_recuperado']) / df_full['c_recuperado']
    df_full = df_full.fillna(0)
    df_entrenamiento = df_full[['fecha', 'casos_diff']].iloc[:int(round((len(df_full) * 0.8) - 1, 0))]
    test = df_entrenamiento.set_index('fecha')
    p_v = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    d_v = [0,1,2]
    q_v = [0,1,2,3,4,5,6,7,8,9,10]
    evaluate_models(test['casos_diff'],p_v,d_v,q_v)
    #plt.plot(df_full['fecha'],df_full['c_caso'])
    # %%

if __name__ == '__main__':
    main()