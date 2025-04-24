import os
import pandas as pd
import zipfile


def unzip_file(zip_path, extract_to='.'):
    """
    Разархивирует ZIP-архив в указанную директорию.
    
    :param zip_path: Путь к ZIP-архиву
    :param extract_to: Директория для извлечения (по умолчанию текущая)
    """
    try:
        # Создать директорию, если она не существует
        os.makedirs(extract_to, exist_ok=True)
        
        # Открыть ZIP-архив
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Извлечь все файлы
            zip_ref.extractall(extract_to)
            print(f'Архив успешно извлечен в: {extract_to}')
            
    except FileNotFoundError:
        print(f'Ошибка: файл {zip_path} не найден.')
    except zipfile.BadZipFile:
        print('Ошибка: файл не является ZIP-архивом или архив поврежден.')
    except Exception as e:
        print(f'Произошла ошибка: {str(e)}')

# Пример использования
#unzip_file('D1.zip', extract_to='D1_data')
#unzip_file('H1.zip', extract_to='H1_data')




def process_daily_data(input_folder, output_folder):
    """
    Обрабатывает CSV-файлы из указанной входной папки, преобразует данные в дневные интервалы
    и сохраняет результаты в выходную папку.
    
    Параметры:
    input_folder (str): Путь к папке с исходными CSV-файлами
    output_folder (str): Путь к папке для сохранения обработанных данных
    """
    # Создаем выходную папку, если она не существует
    os.makedirs(output_folder, exist_ok=True)
    
    # Обрабатываем каждый CSV-файл во входной папке
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            # Читаем файл
            df = pd.read_csv(os.path.join(input_folder, file))
            
            # Преобразуем колонку 'begin' в datetime
            df['begin'] = pd.to_datetime(df['begin'])
            
            # Создаем колонку с датой (без времени)
            df['date'] = df['begin'].dt.date
            
            # Группируем по дате и агрегируем данные
            daily = df.groupby('date').agg({
                'open': 'first',
                'close': 'last',
                'high': 'max',
                'low': 'min',
                'value': 'sum',
                'volume': 'sum',
                'ticker': 'first'
            }).reset_index()
            
            # Сохраняем результат
            output_path = os.path.join(output_folder, file)
            daily.to_csv(output_path, index=False, encoding='utf-8')


def process_folder_data(folder_path, start_date='2010-01-01', threshold=0.01):
    """
    Обрабатывает все CSV-файлы в указанной папке, объединяет данные,
    фильтрует по дате и удаляет столбцы с пропусками.

    Параметры:
    - folder_path: путь к папке с CSV-файлами
    - start_date: начальная дата для фильтрации (по умолчанию '2010-01-01')
    - threshold: максимально допустимая доля пропусков (по умолчанию 0.01)

    Возвращает:
    - Объединенный и очищенный DataFrame
    """
    
    data_frames = []

    # Обработка каждого CSV-файла
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            
            # Загрузка и преобразование данных
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['tic'] = file_name.replace('.csv', '')
            
            # Преобразование в широкий формат
            df_pivot = df.pivot(index='datetime', columns='tic', values=['close','high','low','volume','open'])
            data_frames.append(df_pivot)

    # Объединение данных
    merged_data = pd.concat(data_frames, axis=1)
    
    # Фильтрация по дате
    merged_data = merged_data.loc[start_date:]
    
    # Удаление столбцов с пропусками
    null_percent = merged_data.isna().mean()
    columns_to_drop = null_percent[null_percent >= threshold].index
    merged_data = merged_data.drop(columns=columns_to_drop)

    return merged_data

# Вызов функции
"""
result = process_folder_data(
    folder_path="D1_data",
    start_date='2010-01-01',
    threshold=0.001
)

# Просмотр первых 5 строк
print(result.head())"""