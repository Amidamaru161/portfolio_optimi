from datetime import datetime, timedelta
import os
import sys
import pandas as pd
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor

# Определение параметров DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Создание DAG
dag = DAG(
    'portfolio_optimization',
    default_args=default_args,
    description='Оптимизация инвестиционного портфеля на основе проанализированных данных',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Функция для расчета ковариационной матрицы и корреляций
def calculate_correlations():
    # Загружаем список обработанных тикеров
    output_dir = '/opt/airflow/data_processed/indicators'
    with open(f'{output_dir}/processed_tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f.readlines()]
    
    # Создаем DataFrame для хранения доходностей
    returns_df = pd.DataFrame()
    
    # Загружаем данные по каждому тикеру
    for ticker in tickers:
        try:
            ticker_path = f'{output_dir}/{ticker}_advanced_indicators.csv'
            ticker_df = pd.read_csv(ticker_path, index_col=0)
            ticker_df.index = pd.to_datetime(ticker_df.index)
            
            # Добавляем доходность в общий DataFrame
            returns_df[ticker] = ticker_df['daily_return']
        except Exception as e:
            print(f"Ошибка при обработке тикера {ticker}: {e}")
    
    # Удаляем строки с пропущенными значениями
    returns_df = returns_df.dropna()
    
    # Рассчитываем ковариационную матрицу
    cov_matrix = returns_df.cov()
    
    # Рассчитываем корреляционную матрицу
    corr_matrix = returns_df.corr()
    
    # Создаем директорию для результатов, если она не существует
    result_dir = '/opt/airflow/data_processed/portfolio'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Сохраняем результаты
    cov_matrix.to_csv(f'{result_dir}/covariance_matrix.csv')
    corr_matrix.to_csv(f'{result_dir}/correlation_matrix.csv')
    returns_df.to_csv(f'{result_dir}/returns.csv')

# Функция для оптимизации портфеля по методу Марковица
def optimize_portfolio():
    # Загружаем данные
    result_dir = '/opt/airflow/data_processed/portfolio'
    returns_df = pd.read_csv(f'{result_dir}/returns.csv', index_col=0)
    cov_matrix = pd.read_csv(f'{result_dir}/covariance_matrix.csv', index_col=0)
    
    # Получаем список тикеров
    tickers = returns_df.columns.tolist()
    
    # Рассчитываем средние доходности
    mean_returns = returns_df.mean()
    
    # Определяем количество активов
    num_assets = len(tickers)
    
    # Генерируем случайные портфели для эффективной границы
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        # Генерируем случайные веса
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Рассчитываем доходность портфеля
        portfolio_return = np.sum(mean_returns * weights) * 252  # Годовая доходность
        
        # Рассчитываем волатильность портфеля
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights))) * np.sqrt(252)
        
        # Рассчитываем коэффициент Шарпа (безрисковая ставка = 0)
        sharpe_ratio = portfolio_return / portfolio_volatility
        
        # Записываем результаты
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = sharpe_ratio
    
    # Создаем DataFrame с результатами
    columns = ['Return', 'Volatility', 'Sharpe']
    portfolios = pd.DataFrame(results.T, columns=columns)
    
    # Находим портфель с максимальным коэффициентом Шарпа
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_port = {
        'Return': results[0, max_sharpe_idx],
        'Volatility': results[1, max_sharpe_idx],
        'Sharpe Ratio': results[2, max_sharpe_idx],
        'Weights': dict(zip(tickers, weights_record[max_sharpe_idx]))
    }
    
    # Находим портфель с минимальной волатильностью
    min_vol_idx = np.argmin(results[1])
    min_vol_port = {
        'Return': results[0, min_vol_idx],
        'Volatility': results[1, min_vol_idx],
        'Sharpe Ratio': results[2, min_vol_idx],
        'Weights': dict(zip(tickers, weights_record[min_vol_idx]))
    }
    
    # Сохраняем результаты в CSV
    portfolios.to_csv(f'{result_dir}/efficient_frontier.csv')
    
    # Сохраняем оптимальные портфели
    max_sharpe_weights = pd.DataFrame(list(max_sharpe_port['Weights'].items()), columns=['Ticker', 'Weight'])
    max_sharpe_weights.to_csv(f'{result_dir}/max_sharpe_portfolio.csv', index=False)
    
    min_vol_weights = pd.DataFrame(list(min_vol_port['Weights'].items()), columns=['Ticker', 'Weight'])
    min_vol_weights.to_csv(f'{result_dir}/min_volatility_portfolio.csv', index=False)
    
    # Создаем файл с отметкой о завершении
    with open(f'{result_dir}/optimization_complete.txt', 'w') as f:
        f.write(f"Optimization completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        f.write(f"\nMax Sharpe Ratio Portfolio:\n")
        f.write(f"Expected Return: {max_sharpe_port['Return']:.4f}\n")
        f.write(f"Volatility: {max_sharpe_port['Volatility']:.4f}\n")
        f.write(f"Sharpe Ratio: {max_sharpe_port['Sharpe Ratio']:.4f}\n\n")
        f.write(f"Min Volatility Portfolio:\n")
        f.write(f"Expected Return: {min_vol_port['Return']:.4f}\n")
        f.write(f"Volatility: {min_vol_port['Volatility']:.4f}\n")
        f.write(f"Sharpe Ratio: {min_vol_port['Sharpe Ratio']:.4f}\n")

# Сенсор для ожидания завершения анализа индикаторов
wait_for_analysis = FileSensor(
    task_id='wait_for_analysis',
    filepath='/opt/airflow/data_processed/indicators/analysis_complete.txt',
    poke_interval=60,  # Проверять каждую минуту
    timeout=60 * 60 * 2,  # Таймаут 2 часа
    mode='reschedule',  # Освобождать слот во время ожидания
    dag=dag,
)

# Задача для расчета корреляций и ковариаций
calculate_corr = PythonOperator(
    task_id='calculate_correlations',
    python_callable=calculate_correlations,
    dag=dag,
)

# Задача для оптимизации портфеля
optimize_port = PythonOperator(
    task_id='optimize_portfolio',
    python_callable=optimize_portfolio,
    dag=dag,
)

# Определение порядка выполнения задач
wait_for_analysis >> calculate_corr >> optimize_port 