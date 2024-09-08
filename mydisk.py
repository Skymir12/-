# -*- coding: utf-8 -*-

from sys import argv
import datetime


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
			print(f'\n[?] Помошь по пользованию программой mydisk...\n{str_text_app_name}: использование: python3 {str_text_app_name} <ваш файл для анализа>\n')
		else:
			current_date = datetime.date.today().isoformat()
			print(f'\n[*] Запуск сценария программы...\n{str_text_app_name}: исходный файл для анализа загружен: {str_text_input_file}\n{len(str_text_app_name)*' '}  результат анализа будет сохранен в файл: outdata_{str_text_input_file}\n{len(str_text_app_name)*' '}  логи и ход выполнения программы будет сохранен в файл: .logs_{current_date}_{str_text_input_file}\n')

	except:
		print(f'\n[!] Ошибка сценария программы...\n{str_text_app_name}: ошибка: введены пустые или неверные аргументы\n{len(str_text_app_name)*' '}  проверьте корректность вводимых данных или воспользуйтесь для помощи аргументом: "?"\n')


util()
