import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier



X_train, y_train, X_test, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

model = LGBMClassifier()

# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)

params = {
    'n_estimators': [50, 100, 120, 170],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'num_leavels': [50, 100, 200, 400]
}

grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, scoring='accuracy')

grid_search.fit(X_train, y_train, )
print(f"Параметры: {grid_search.best_params_}")

best_model = grid_search.best_estimator_

