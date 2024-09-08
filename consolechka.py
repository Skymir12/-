# -*- coding: utf-8 -*-

from sys import argv
import datetime
import os
import lightgbm

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier

try:
   import pandas as pd
   import pickle
except:
   print('setup start...')
   try:
      os.system('pip install pandas pickle')
   except Exception as e:
      print(f'error: {e}')
   print('setup end')

banner = '''
  _________________                        _ _     _
 |# :           : #|                      | (_)   | |
 |  :           :  |   _ __ ___  _   _  __| |_ ___| | __
 |  :           :  |  | '_ ` _ \\| | | |/ _` | / __| |/ /
 |  :           :  |  | | | | | | |_| | (_| | \\__ \\   <
 |  :___________:  |  |_| |_| |_|\\__, |\\__,_|_|___/_|\\_\\
 |     _________   |              __/ |
 |    | __      |  |             |___/
 |    ||  |     |  |
 \\____||__|_____|__|                           by nkeivt
'''

def util():
    str_text_app_name = str(argv[0])

    try:
        str_text_input_file = str(argv[1])
        if str_text_input_file == '?':
            print(banner)
            print(f'\n[?] Помощь по пользованию программой mydisk...\n{str_text_app_name}: использование: python3 {str_text_app_name} <ваш файл для анализа>\n')
        else:
            current_date = datetime.date.today().isoformat()
            print(f'\n[*] Запуск сценария программы...\n{str_text_app_name}: исходный файл для анализа загружен: {str_text_input_file}\n')

            # Load input dataset
            df = pd.read_csv(str_text_input_file)

            # Encoding 'model' and 'serial_number'
            df['model_encode'] = df['model'].astype('category').cat.codes
            df['serial_number_encode'] = df['serial_number'].astype('category').cat.codes

            # Select necessary columns
            df_input = df[['model_encode', 'serial_number_encode', 'capacity_bytes', 'failure']]

            # Load pre-trained mode
            try:
                with open("estimator.pkl", "rb") as f:
                    model = pickle.load(f)
            except Exception as e:
                print(e)


            # Make predictions
            try:
                predictions = model.predict(df_input)
            except:
                print('err')

            # Save the analysis results
            try:
                result_file = f'outdata_{str_text_input_file}'
                df_input['predictions'] = predictions
                df_input.to_csv(result_file, index=False)
                print(f'Results saved to {result_file}')
            except:
                print('err')

            # Create logs
            log_file = f'.logs_{current_date}_{str_text_input_file}'
            with open(log_file, 'w') as log:
                log.write(f'Input file: {str_text_input_file}\n')
                log.write(f'Results file: {result_file}\n')
                log.write(f'Log date: {current_date}\n')
            print(f'Logs saved to {log_file}')

    except Exception as e:
        print(f'\n[!] Ошибка сценария программы...\n{str_text_app_name}: ошибка: {e}\n')
        print(f'{str_text_app_name}: проверьте корректность вводимых данных или воспользуйтесь для помощи аргументом: "?"\n')

if __name__ == "__main__":
    util()
