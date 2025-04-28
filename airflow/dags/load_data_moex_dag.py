from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import pandas as pd
import time
import os

from utils.load_data import  download_moex_candles
from utils.hour_to_daily import process_daily_data
 
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}




with DAG(
    'moex_daily_pipeline',
    default_args=default_args,
    description='Полный пайплайн загрузки и обработки данных MOEX',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['moex', 'processing'],
) as dag:

    download_task = PythonOperator(
        task_id='download_candles',
        python_callable=download_moex_candles,
        op_kwargs={
            'interval': 60,
            'limit': 500,
            'save_path': '/opt/airflow/data/moex/',
            'delay': 0.1
        }
    )

    process_task = PythonOperator(
        task_id='process_daily_data',
        python_callable=process_daily_data,
        op_kwargs={
            'input_folder': '/opt/airflow/data/moex/',
            'output_folder': '/opt/airflow/data/moex_daily/'
        }
    )

    download_task >> process_task