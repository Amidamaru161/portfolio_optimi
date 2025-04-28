import requests
import pandas as pd
import time
import os
def download_moex_candles(**kwargs):
    """
    Оберточная функция для использования в PythonOperator
    """
    sec_ids = kwargs.get('sec_ids', [
        'ABIO', 'AFKS', 'AFLT', 'AKRN', 'AMEZ', 'APTK', 'BLNG', 'BSPB', 'CHMF', 'CHMK',
        'ELFV', 'FEES', 'FESH', 'GAZP', 'GCHE', 'HYDR', 'INGR', 'IRAO', 'IRKT', 'KMAZ',
        'LKOH', 'LNZLP', 'LNZL', 'LSRG', 'MAGN', 'MGNT', 'MRKC', 'MRKK', 'MRKP', 'MRKU',
        'MRKV', 'MRKY', 'MRKZ', 'MSNG', 'MSRS', 'MTLR', 'MTSS', 'MVID', 'NKNCP', 'NKNC',
        'NLMK', 'NMTP', 'NVTK', 'OGKB', 'PIKK', 'PLZL', 'RASP', 'ROSN', 'RTKMP', 'RTKM',
        'SBERP', 'SBER', 'SNGSP', 'SNGS', 'SVAV', 'TATNP', 'TATN', 'TGKA', 'TGKB', 'TRMK',
        'TRNFP', 'VSMO', 'VTBR'
    ])
    
    interval = kwargs.get('interval', 60)
    limit = kwargs.get('limit', 500)
    save_path = kwargs.get('save_path', '/opt/airflow/data/moex/')  # Используйте абсолютный путь
    delay = kwargs.get('delay', 0.01)

    os.makedirs(save_path, exist_ok=True)

    for sec_id in sec_ids:
        url = f'https://iss.moex.com/iss/engines/stock/markets/shares/securities/{sec_id}/candles.json'
        params = {
            'start': '0',
            'interval': str(interval),
            'limit': str(limit),
        }
        print(f'START {sec_id}')
        start_time = time.time()
        df = pd.DataFrame()

        while True:
            try:
                response = requests.get(url=url, params=params)
                if response.status_code != 200:
                    print(f'Ошибка {response.status_code} для {sec_id}')
                    break
                data = response.json()
                if 'candles' not in data or not data['candles']['data']:
                    break
                temp_df = pd.DataFrame(data['candles']['data'], columns=data['candles']['columns'])
                df = pd.concat([df, temp_df])
                params['start'] = str(int(params['start']) + int(params['limit']))
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                print(f'Сетевая ошибка при загрузке {sec_id}: {e}')
                break
            except Exception as e:
                print(f'Ошибка при обработке данных {sec_id}: {e}')
                break

        if not df.empty:
            df['ticker'] = sec_id
            file_path = os.path.join(save_path, f'{sec_id}.csv')
            df.to_csv(file_path, index=False, encoding='UTF-8')
            end_time = time.time()
            print(f'Time spent = {round(end_time - start_time, 2)} s')
            print(f'File size = {os.stat(file_path).st_size} bytes')
        else:
            print(f'Нет данных для {sec_id}')

        print(f'END {sec_id}')
