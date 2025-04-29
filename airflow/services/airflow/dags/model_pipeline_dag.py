from datetime import datetime, timedelta
import os
import pandas as pd
import torch
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.preprocessing import MaxAbsScaler

from utils.ml.models import EIIE
from utils.ml.dlr_agent import DRLAgent
from utils.ml.portfolio_env import PortfolioOptimizationEnv
from utils.ml.feature_creating import GroupByScaler
from utils.ml.config import TICKER_LIST, TIME_WINDOW, BATCH_SIZE, EPOCH

# Настройка логгера
logger = logging.getLogger(__name__)


def choose_data(folder_path, date_col='date', start_date='2010-01-01', tic_list=TICKER_LIST):
    """
    Обрабатывает CSV-файлы в указанной папке, объединяет данные,
    фильтрует по дате и удаляет столбцы с пропусками.

    Параметры:
    - folder_path: путь к папке с CSV-файлами
    - date_col: название колонки с датой (по умолчанию 'date')
    - start_date: начальная дата для фильтрации (по умолчанию '2010-01-01')
    - threshold: максимально допустимая доля пропусков (по умолчанию 0.01)
    - tic_list: список тикеров для обработки. Если None, обрабатываются все файлы

    Возвращает:
    - Объединенный и очищенный DataFrame
    """
    logger.info(f"Начало обработки данных из {folder_path}")
    
    data_frames = []

    # Обработка каждого CSV-файла
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            tic = file_name.replace('.csv', '')
            
            # Пропустить тикеры не из списка
            if tic_list is not None and tic not in tic_list:
                continue
                
            file_path = os.path.join(folder_path, file_name)
            logger.debug(f"Обработка файла {file_path}")
            
            # Загрузка и преобразование данных
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df[date_col])
            df['tic'] = tic  # Используем предварительно вычисленное значение
            
            # Преобразование в широкий формат
            df_pivot = df.pivot(index='datetime', columns='tic', values=['close', 'high', 'low', 'open', 'volume'])
            data_frames.append(df_pivot)

    logger.info(f"Объединение {len(data_frames)} обработанных файлов")
    merged_data = pd.concat(data_frames, axis=1)
    
    # Фильтрация по дате
    merged_data = merged_data.loc[start_date:]
    merged_data = merged_data.fillna(method='ffill')
    merged_data = merged_data.stack(level='tic')
    merged_data = merged_data.reset_index()
    
    merged_data['datetime'] = pd.to_datetime(merged_data['datetime'])
    logger.info(f"Обработка данных завершена, получено {len(merged_data)} строк")

    return merged_data


def create_env(data):
    """
    Создает среду для тестирования модели портфельной оптимизации.
    
    Параметры:
    - data: DataFrame с данными
    
    Возвращает:
    - environment_test: среда для тестирования
    - portfolio_norm_df: нормализованные данные
    """
    logger.info("Создание тестовой среды")
    valid_columns = ['open', 'high', 'low', 'close', 'volume', 'tic'] 
    gbsclr = GroupByScaler(by="tic", scaler=MaxAbsScaler)
    logger.debug("Нормализация данных")
    portfolio_norm_df = gbsclr.fit_transform(data[valid_columns])
    portfolio_norm_df['date'] = data['datetime']

    logger.debug("Инициализация среды PortfolioOptimizationEnv")
    environment_test = PortfolioOptimizationEnv(
        portfolio_norm_df,
        initial_amount=1000000,
        comission_fee_pct=0.0025,
        reward_scaling=0.01,
        time_window=TIME_WINDOW,
        features=["close", "high", "low", "volume", "open"],
        normalize_df=None
    )   
    logger.info("Тестовая среда создана успешно")
    return environment_test, portfolio_norm_df


def evaluate_model(**context):
    """
    Оценивает производительность обученной модели.
    
    Параметры:
    - data_folder: путь к папке с тестовыми данными
    - model_path: путь к сохраненной модели
    - output_path: путь для сохранения результатов
    """
    logger.info("Начало оценки модели")
    data_folder = context['params'].get('data_folder', '/opt/airflow/data/moex_daily/')
    model_path = context['params'].get('model_path', 
                                      f"/opt/airflow/models/policy_EIIE_batch_{BATCH_SIZE}_window_{TIME_WINDOW}_epoch_{EPOCH}.pt")
    output_path = context['params'].get('output_path', '/opt/airflow/results/')
    
    logger.info(f"Параметры: data_folder={data_folder}, model_path={model_path}, output_path={output_path}")
    
    # Подготовка данных
    logger.info("Загрузка и подготовка данных")
    data = choose_data(data_folder)
    environment_test, _ = create_env(data)
    
    # Загрузка модели
    logger.info("Загрузка модели")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используемое устройство: {device}")
    policy = EIIE(time_window=TIME_WINDOW, device=device, initial_features=5, k_size=4)
    try:
        policy.load_state_dict(torch.load(model_path))
        logger.info(f"Модель успешно загружена из {model_path}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise
    
    # Создание модели для тестирования
    logger.info("Инициализация агента DRL")
    model = DRLAgent(environment_test).get_model("pg", device)
    
    # Оценка модели
    logger.info("Начало процесса валидации")
    EIIE_results = {"test": {}}
    try:
        DRLAgent.DRL_validation(model, environment_test, policy=policy)
        EIIE_results["test"]["value"] = environment_test._asset_memory["final"]
        logger.info(f"Валидация завершена. Результат: {EIIE_results}")
    except Exception as e:
        logger.error(f"Ошибка при валидации: {str(e)}")
        raise
    
    # Сохранение результатов
    os.makedirs(output_path, exist_ok=True)
    result_file = os.path.join(output_path, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Результаты тестирования: {EIIE_results}")
    logger.info(f"Результаты сохранены в {result_file}")
    
    return f"Результаты сохранены в {result_file}"


# Настройки DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=20),
}

# Определение DAG
with DAG(
    'portfolio_optimization_evaluation',
    default_args=default_args,
    description='Пайплайн для оценки модели портфельной оптимизации на новых данных',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['portfolio', 'optimization', 'evaluation'],
) as dag:

    # Задача оценки модели
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        params={
            'data_folder': '/opt/airflow/data/moex_daily/',
            'model_path': f"/opt/airflow/models/policy_EIIE_batch_{BATCH_SIZE}_window_{TIME_WINDOW}_epoch_{EPOCH}.pt",
            'output_path': '/opt/airflow/results/'
        },
        provide_context=True
    )