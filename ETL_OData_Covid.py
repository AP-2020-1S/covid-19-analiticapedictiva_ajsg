import requests
import pandas as pd
import time

# %%
# filters = 'fecha_de_notificaci_n=2020-08-17T00:00:00.000'

def extract_data(api_url, limit):
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

def data_quality(df):
    if


# %%
def main():
    # %%
    api_url = "https://www.datos.gov.co/resource/gt2j-8ykr.json"
    df_casos = extract_data(api_url, 50000)
    # %%

if __name__ == '__main__':
    main()