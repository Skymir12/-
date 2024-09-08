#!/usr/bin/env python
# coding: utf-8

# Импорт библиотек

# In[264]:


import pandas as pd 
import os
import numpy as np
import datetime
import math


# In[3]:


df = pd.read_csv("C:/Users/user/Desktop/ds/data_Q1_2023/2023-01-01.csv")


# In[44]:


pd.options.mode.chained_assignment = None


# In[9]:


os.listdir("C:/Users/user/Desktop/ds/data_Q1_2023/")


# In[23]:


df_last = pd.DataFrame()


# In[7]:


dirlist = os.listdir("C:/Users/user/Desktop/ds/")


# Выбираем только сломанные диски

# In[26]:


for folders in dirlist:
    path = "C:/Users/user/Desktop/ds/"+folders
    file_list = os.listdir(path)
    for file in file_list:
        try:
            df_temp = pd.read_csv("C:/Users/user/Desktop/ds/"+folders+"/"+file)
            df_temp = df_temp.loc[df_temp['failure'] == 1]
            df_last = pd.concat([df_last, df_temp])
        except Exception as e:
            print(e)


# Выбираем абсолютно все диски

# In[61]:


df_all = pd.DataFrame()


# In[62]:


for folders in dirlist:
    path = "C:/Users/user/Desktop/ds/"+folders
    file_list = os.listdir(path)
    for file in file_list:
        try:
            df_temp = pd.read_csv("C:/Users/user/Desktop/ds/"+folders+"/"+file)
            df_all = pd.concat([df_all, df_temp])
            df_all = df_all.drop_duplicates(subset="serial_number", keep="first")
        except Exception as e:
            print(e)


# In[ ]:





# In[63]:


df_all


# Сохранение решения

# In[158]:


df_all.to_csv("C:/Users/user/Desktop/ds_all.csv", index=False)


# In[27]:


df_last


# In[30]:


df_last.to_csv("C:/Users/user/Desktop/df_final.csv", index=False)


# In[ ]:





# In[68]:


df_all["capacity_bytes"].value_counts()


# In[ ]:





# Удаление выбросов
# 
# 

# In[72]:


df_all


# Заменяем все записи с capacity_bytes равной с -1 на 0

# In[6]:


df_all['capacity_bytes'].mask(df_all['capacity_bytes'] == -1, 0, inplace=True)


# In[7]:


df_all["capacity_bytes"].value_counts()


# In[8]:


df_all_counts = df_all["capacity_bytes"].value_counts()
mask_df = df_all_counts[df_all_counts > 10].index


# In[9]:


df_test_all = df_all[df_all["capacity_bytes"].isin(mask_df)]


# In[10]:


df_test_all


# Результат очистки

# In[11]:


df_test_all["capacity_bytes"].value_counts()


# Создаём столбец с конечной датой

# In[ ]:


df_all["date_end"] = ""
df_all["month_life"] = ""


# In[24]:


for ind in df_all.index:
    try:
        if len(df_last.loc[df_last['serial_number'] == df_all["serial_number"][ind]]) > 0:
            df_t = df_last.loc[df_last['serial_number'] == df_all["serial_number"][ind]]
            df_all.loc[ind, "date_end"] = df_all.loc[ind, "date_end"].replace("", df_t.iloc[0]["date"])
        else:
            df_all.loc[ind, "date_end"] = "2024-06-30"
    except Exception as e:
        print(e)
        break


# In[107]:


df_all["month_life"].value_counts()


# Функция для правильной конвертации месяцев для лучшей эффективности

# In[142]:


def convert_months(month):
    tp = {3: [0,1,2,3], 6: [4,5,6], 9:[7,8,9], 12: [10, 11, 12], 15: [13,14,15], 18: [16, 17, 18]}
    last_key = 0
    for key, value in tp.items():
        last_key = key
        month = int(month)
        if month > 3 and month in value and month != key:
            return int(key-3)
        elif month > 3 and month in value and month == key:
            return int(key)
        elif month < 3:
            return int(key)


# In[154]:


df_all['month_life'] = ((df_all["date_end"] - df_all["date"]) / np.timedelta64(1, 'D'))/30


# Конвертация из текста в тип данных времени

# In[84]:


df_all["date_end"] = pd.to_datetime(df_all["date_end"])
df_all["date"] = pd.to_datetime(df_all["date"])


# In[152]:


df_all


# Умная конвертация месяцев

# In[156]:


df_all["month_life"] = df_all["month_life"].apply(lambda x: convert_months(x))


# In[157]:


df_all["month_life"].value_counts()


# In[ ]:





# In[ ]:





# Загрузка датасетов

# In[20]:


df_all = pd.read_csv("C:/Users/user/Desktop/ds_all.csv")
df_last = pd.read_csv("C:/Users/user/Desktop/df_final.csv")


# In[ ]:





# In[159]:


df_all.fillna(0)


# In[ ]:


df_all['smart_normalized'] = df_all['consumer_budg'] + df_all['enterprise_budg']


# In[ ]:





# In[163]:


df_all


# In[ ]:


df[""


# In[ ]:





# In[173]:


anton_df = pd.read_csv("C:/Users/user/Desktop/Anton_ds_all.csv")


# In[174]:


anton_df


# In[486]:


df = pd.DataFrame()
df2 = pd.DataFrame()


# In[487]:


df["model_encode"] = anton_df["model_encode"]


# In[488]:


df["serial_number_encode"] = anton_df["serial_number_encode"]


# In[489]:


df["capacity_bytes"] = anton_df["capacity_bytes"]


# In[490]:


df["failure"] = anton_df["failure"]


# In[491]:


df["month_life"] = anton_df["month_life"]


# In[492]:


df = df.sample(frac=1)


# In[ ]:





# In[493]:


import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier


# In[494]:


X = np.array(df.drop("month_life", axis=1))
y = np.array(df["month_life"])
#Фиксим наны
y = np.array(y)
y = y.astype(int)


# In[495]:


model = LGBMClassifier()
params = {
    'n_estimators': [50, 100, 120, 170],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'num_leaves': [50, 100, 200, 400]
}
grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, scoring='accuracy')


# In[496]:


grid_search.fit(X[0:10000], y[0:10000])


# In[498]:


est = grid_search.best_estimator_


# In[504]:


est.fit(X[0:10000], y[0:10000])


# In[505]:


est.predict(X[270000:270200])


# In[502]:


y[270000:270200]


# In[479]:


y = np.array(y)


# In[506]:


import pickle
with open("C:/Users/user/Desktop/estimator.pkl", "wb") as f:
    pickle.dump(est, f)


# In[ ]:


with open("C:/Users/user/Desktop/estimator.pkl", "rb") as f:
    model = pickle.load(f)


# In[318]:


df2


# In[480]:


y = y.astype(int)


# In[377]:


train_dataset = lightgbm.Dataset(X_train, X_train)
test_dataset = lightgbm.Dataset(X_test, y_test)


# In[ ]:





# In[451]:





# In[ ]:





# In[ ]:





# In[ ]:




