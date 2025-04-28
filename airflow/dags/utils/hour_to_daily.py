
import pandas as pd
import os

def process_daily_data(**kwargs):
    """
    Обработка данных через PythonOperator
    """
    input_folder = kwargs.get('input_folder', '/opt/airflow/data/moex/')
    output_folder = kwargs.get('output_folder', '/opt/airflow/data/moex_daily/')
    
    os.makedirs(output_folder, exist_ok=True)
    
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(input_folder, file))
                
                # Обработка данных
                df['begin'] = pd.to_datetime(df['begin'])
                df['date'] = df['begin'].dt.date
                
                daily = df.groupby('date').agg({
                    'open': 'first',
                    'close': 'last',
                    'high': 'max',
                    'low': 'min',
                    'value': 'sum',
                    'volume': 'sum',
                    'ticker': 'first'
                }).reset_index()
                
                # Сохранение результата
                output_path = os.path.join(output_folder, file)
                daily.to_csv(output_path, index=False, encoding='utf-8')
                print(f'Processed {file} successfully')
                
            except Exception as e:
                print(f'Error processing {file}: {str(e)}')
                continue