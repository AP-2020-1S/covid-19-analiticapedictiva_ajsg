#
# Escalamiento de los datos
#  El StandardScaler remueve la media de los datos y
#  luego los divide por su desviación estándar
#
from sklearn.preprocessing import StandardScaler


# Importar paquetes requeridos
import requests
import pandas as pd
from datetime import date
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
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
            df['extracted_at_utc'] = pd.to_datetime('now', utc=True)
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
    df_in['fecha_diagnostico'] = pd.to_datetime(df_in['fecha_diagnostico'])
    return df_in, df_dates


def data_agg_col(df, df_dates, city_filter):
    # Funcion para realizar agregaciones
    if city_filter is None:
        df_filtered = df
    else:
        df_filtered = df.loc[df['ciudad_de_ubicaci_n'].isin(city_filter)]
    df_filtered['c_recuperado'] = df_filtered['atenci_n'].apply(lambda x : 1 if x == 'recuperado' else 0)
    df_filtered['c_fallecido'] = df_filtered['atenci_n'].apply(lambda x: 1 if x == 'fallecido' else 0)
    df_filtered['c_caso'] = 1
    #df_final = df.groupby(['fecha_diagnostico'], as_index=False).agg({'id_de_caso': 'count', 'c_recuperado': 'sum', 'c_fallecido': 'sum'})
    df_casos = df_filtered.groupby(['fecha_diagnostico'], as_index=False)['c_caso'].sum()
    df_casos = df_casos.rename(columns={'fecha_diagnostico': 'fecha'})
    df_muertes = df_filtered.groupby(['fecha_de_muerte'], as_index=False)['c_fallecido'].sum()
    df_muertes = df_muertes.rename(columns={'fecha_de_muerte':'fecha'})
    df_recuperados = df_filtered.groupby(['fecha_recuperado'], as_index=False)['c_recuperado'].sum()
    df_recuperados = df_recuperados.rename(columns={'fecha_recuperado': 'fecha'})
    #df_final = df_final.rename(columns={'id_de_caso':'total_casos','c_recuperado':'total_recuperados','c_fallecido':'total_fallecidos'})
    df_full = pd.merge(df_dates, df_casos, how='left', on=['fecha'])
    df_full = pd.merge(df_full, df_muertes, how='left', on=['fecha'])
    df_full = pd.merge(df_full, df_recuperados, how='left', on=['fecha'])
    return df_full


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order, train_perc):
    # prepare training dataset
    train_size = int(len(X) * train_perc)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(arima_order))
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
                    mse = evaluate_arima_model(dataset, order, 0.8)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


def data_prep_normalize(df_o):
    df = df_o.copy()
    df['casos_lag'] = df['c_caso'].shift(1)
    df['fallecidos_lag'] = df['c_fallecido'].shift(1)
    df['recuperados_lag'] = df['c_recuperado'].shift(1)
    df = df.replace({0: np.nan})
    df['casos_diff'] = (df['c_caso'] - df['casos_lag'])/df['casos_lag']
    df['fallecidos_diff'] = (df['c_fallecido'] - df['fallecidos_lag']) / df['fallecidos_lag']
    df['recuperados_diff'] = (df['c_recuperado'] - df['recuperados_lag']) / df['recuperados_lag']
    df['tasa_casos'] = df['casos_lag']/df['c_caso']
    df['tasa_fallecitos'] = df['fallecidos_lag']/df['c_fallecido']
    df['tasa_recuperados'] = df['recuperados_lag']/ df['c_recuperado']

    # Eliminacion de Outliers
    quant_95_casos = df['casos_diff'].quantile(.95)
    mean_casos = df['casos_diff'].mean()

    quant_95_fallecidos = df['fallecidos_diff'].quantile(.95)
    mean_fallecidos = df['fallecidos_diff'].mean()

    quant_95_recuperados = df['recuperados_diff'].quantile(.95)
    mean_recuperados = df['recuperados_diff'].mean()

    df['casos_diff'] = df['casos_diff'].apply(lambda x: x if x <= quant_95_casos else mean_casos)
    df['fallecidos_diff'] = df['fallecidos_diff'].apply(lambda x: x if x <= quant_95_fallecidos else mean_fallecidos)
    df['recuperados_diff'] = df['recuperados_diff'].apply(lambda x: x if x <= quant_95_recuperados else mean_recuperados)

    df = df.fillna(0)
    # Eliminacion de outliers ultimas fechas por datos faltantes desde la fuente
    df.drop(df.tail(2).index, inplace=True)
    return df


def arima_model_imp(df, train_perc, pred_range, ranges_casos, ranges_recu, ranges_fall):
    df_entrenamiento = df[['fecha', 'casos_diff']].iloc[:int(round((len(df) * train_perc) - 1, 0))]
    df_entrenamiento_recu = df[['fecha', 'recuperados_diff']].iloc[:int(round((len(df) * train_perc) - 1, 0))]
    df_entrenamiento_fall = df[['fecha', 'fallecidos_diff']].iloc[:int(round((len(df) * train_perc) - 1, 0))]
    serie_entrenamienti = df_entrenamiento.set_index('fecha')
    serie_entrenamienti_recu = df_entrenamiento_recu.set_index('fecha')
    serie_entrenamienti_fall = df_entrenamiento_fall.set_index('fecha')

    model_casos = ARIMA(serie_entrenamienti['casos_diff'], order=(ranges_casos))
    model_fit_casos = model_casos.fit(disp=0)
    model_recu = ARIMA(serie_entrenamienti_recu['recuperados_diff'], order=(ranges_recu))
    model_fit_recu = model_recu.fit(disp=0)
    model_fall = ARIMA(serie_entrenamienti_fall['fallecidos_diff'], order=(ranges_fall))
    model_fit_fall = model_fall.fit(disp=0)
    # Pronosticos corto plazo
    yhat_corto_casos = model_fit_casos.forecast(steps=pred_range[0])[0]
    yhat_corto_recuperados = model_fit_recu.forecast(steps=pred_range[0])[0]
    yhat_corto_fallecidos = model_fit_fall.forecast(steps=pred_range[0])[0]
    # Pronosticos largo plazo
    yhat_largo_casos = model_fit_casos.forecast(steps=pred_range[1])[0]
    yhat_largo_recuperados = model_fit_recu.forecast(steps=pred_range[1])[0]
    yhat_largo_fallecidos = model_fit_fall.forecast(steps=pred_range[1])[0]
    response = {
        'pronostico_casos': {'corto_plazo': yhat_corto_casos, 'largo_plazo': yhat_largo_casos},
        'pronostico_recuperados': {'corto_plazo': yhat_corto_recuperados, 'largo_plazo': yhat_largo_recuperados},
        'pronostico_fallecidos': {'corto_plazo': yhat_corto_fallecidos, 'largo_plazo': yhat_largo_fallecidos}
         }
    return response


def error_calculation(df, train_perc,range_c, range_rec, range_fall):
    serie_casos = df.set_index('fecha')
    serie_recuperados = df.set_index('fecha')
    serie_fallecidos = df.set_index('fecha')
    error_casos = evaluate_arima_model(serie_casos['casos_diff'], range_c, train_perc)
    error_recu = evaluate_arima_model(serie_recuperados['recuperados_diff'], range_rec, train_perc)
    error_fall = evaluate_arima_model(serie_fallecidos['fallecidos_diff'], range_fall, train_perc)
    response = [
        {'error_casos': error_casos},
        {'error_recuperados': error_recu},
        {'error_fallecidos': error_fall}

    ]
    return response


def translate_prediction(last_value_base, y_pred):
    temp_val = last_value_base
    trans_values = []
    for val in y_pred:
        new_t_val = temp_val*(1 + val)
        trans_values.append(new_t_val)
        temp_val = new_t_val
    return trans_values

# %%


def main():
    # %%
    api_url = "https://www.datos.gov.co/resource/gt2j-8ykr.json"
    df_casos = extract_data(api_url, 50000)
    df_casos, df_dates = data_transform(df_casos, date(2020, 3, 1))
    df_full = data_agg_col(df_casos, df_dates, None)
    df_full_medellin = data_agg_col(df_casos, df_dates, ['Medellín'])
    df_full_bogota = data_agg_col(df_casos, df_dates, ['Bogotá D.C.'])
    df_full_cartagena = data_agg_col(df_casos, df_dates, ['Cartagena de Indias'])
    df_full_cali = data_agg_col(df_casos, df_dates, ['Cali'])
    df_full_barranquilla = data_agg_col(df_casos, df_dates, ['Barranquilla'])
    # Visualizar la distribucion de los datos graficamente previo a la normalizacion
    # plt.plot(df_full['fecha'],df_full['c_caso'])
    df_norm_col = data_prep_normalize(df_full)
    df_norm_medellin = data_prep_normalize(df_full_medellin)
    df_norm_bogota = data_prep_normalize(df_full_bogota)
    df_norm_cartagena = data_prep_normalize(df_full_cartagena)
    df_norm_cali = data_prep_normalize(df_full_cali)
    df_norm_barranquilla = data_prep_normalize(df_full_barranquilla)
    # Visualizar la distribucion de los datos graficamente posterior a la normalizacion
    # plt.plot(df_full['fecha'],df_full['casos_diff'])
    # Implementacion ARIMA
    # Bloque validacion del modelo y deteccion p, d, q optimo (Solo aplicar para re calibrar el modelo)
    '''p = [0,1,3]
    d = [0,1]
    q = [0,1,3]
    df_entrenamiento = df_norm_medellin[['fecha', 'fallecidos_diff']].iloc[:int(round((len(df_norm_medellin) * 0.8) - 1, 0))]
    test = df_norm_cali.set_index('fecha')
    evaluate_models(test['casos_diff'], p, d, q)'''
    ###
    # Colombia casos [4, 0, 0] MSE=0.082
    # Colombia Recuperados [1, 0, 3] MSE=0.024
    # Colombia Fallecidos [1, 1, 3] MSE=0.007

    # Medellin casos [4, 1, 2] MSE=0.065
    # Medellin recuperados [2, 1, 2] MSE=0.084
    # Medellin Fallecidos [0, 0, 2] MSE=0.158
    # Cali casos [6, 1, 0] MSE=0.621
    # Cali recuperados [2, 1, 4] MSE=0.138
    # Cali fallecidos [6, 1, 0] MSE=0.068
    # Barranquilla casos [4, 0, 0] MSE=0.430
    # Barranquilla recuperados [0, 0, 3] MSE=0.275
    # Barranquilla Fallecidos [0, 0, 6] MSE=0.940
    # Cartagena Casos [4, 0, 3] MSE=1.043
    # Cartagena recuperados [2, 0, 0] MSE=0.327
    # Cartagena fallecidos [2, 0, 0] MSE=0.788

    response_col = arima_model_imp(df_norm_col, 0.8, [7, 30], [4, 0, 0], [3, 1, 0], [1, 1, 3])

    response_med = arima_model_imp(df_norm_medellin, 0.8, [7, 15], [4, 1, 2], [2, 1, 2], [0, 0, 2])
    response_bog = arima_model_imp(df_norm_bogota, 0.8, [7, 15], [0, 0, 2], [0, 0, 2], [6, 1, 2])
    response_cal = arima_model_imp(df_norm_cali, 0.8, [7, 15], [6, 1, 0], [2, 1, 4], [6, 1, 0])
    response_car = arima_model_imp(df_norm_cartagena, 0.8, [7, 15], [4, 0, 3], [2, 0, 0], [2, 0, 0])
    response_bar = arima_model_imp(df_norm_barranquilla, 0.8, [7, 15], [4, 0, 0], [2, 0, 0], [2, 0, 0])

    last_real_val_casos_col = df_norm_col.tail(1)['c_caso'].values[0]
    last_real_val_rec_col = df_norm_col.tail(1)['c_recuperado'].values[0]
    last_real_val_fall_col = df_norm_col.tail(1)['c_fallecido'].values[0]

    response_col['pronostico_casos']['corto_plazo_translate'] = translate_prediction(last_real_val_casos_col, response_col['pronostico_casos']['corto_plazo'])
    response_col['pronostico_recuperados']['corto_plazo_translate'] = translate_prediction(last_real_val_rec_col, response_col['pronostico_recuperados']['corto_plazo'])
    response_col['pronostico_fallecidos']['corto_plazo_translate'] = translate_prediction(last_real_val_fall_col, response_col['pronostico_fallecidos']['corto_plazo'])

    response_col['pronostico_casos']['largo_plazo_translate'] = translate_prediction(last_real_val_casos_col, response_col['pronostico_casos']['largo_plazo'])
    response_col['pronostico_recuperados']['largo_plazo_translate'] = translate_prediction(last_real_val_rec_col, response_col['pronostico_recuperados']['largo_plazo'])
    response_col['pronostico_fallecidos']['largo_plazo_translate'] = translate_prediction(last_real_val_fall_col, response_col['pronostico_fallecidos']['largo_plazo'])



    # %%

if __name__ == '__main__':
    main()

   # xanterior * (1 + pron_actual)




