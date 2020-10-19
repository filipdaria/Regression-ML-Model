import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def regression_model(model):
    regressor = model
    regressor.fit(X_train_price, Y_train_price)
    score = regressor.score(X_train_price, Y_train_price)
    return regressor, score

cars = pd.read_csv('unclean_focus.csv')
pd.set_option("display.max_rows", None, 'display.max_columns', None)

cars['mileage'].fillna(cars['mileage2'], inplace=True)
cars['fuel type'].fillna(cars['fuel type2'], inplace=True)

petrol = dict.fromkeys(('6', '7', '8', '9', '10', '11', '12', '13', '14', '31', '33', '34', '35', '36', '38', '40'), 'Petrol')
diesel = dict.fromkeys(('15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'), 'Diesel')
cars['fuel type'] = cars['fuel type'].replace(petrol)
cars['fuel type'] = cars['fuel type'].replace(diesel)
cars.drop('mileage2', axis=1, inplace=True)
cars.drop('reference', axis=1, inplace=True)
cars.drop('fuel type2', axis=1, inplace=True)

median_engine_size = cars['engine size2'].dropna().median()

cars.insert(4, 'new col', 'any')
verifier = cars['engine size2'].notnull()
for i in range(len(cars['engine size2'])):
    if not verifier[i]:
        cars['engine size'][i] = cars['engine size2'][i]

for i in range(len(cars['engine size2'])):
    if (float(cars['engine size'][i]) >= float(3)) and (float(cars['engine size2'][i]) >= float(3)):
        cars['engine size'][i] = median_engine_size
    elif (float(cars['engine size2'][i]) >= float(3)) and (float(cars['engine size2'][i]) <= float(3)):
        cars['engine size'][i] = cars['engine size2'][i]

for i in range(len(cars['engine size2'])):
    cars['new col'] = cars[['engine size', 'engine size2']].min(axis=1)

del cars['engine size']
del cars['engine size2']
cars.rename(columns={'new col':'engine size'}, inplace=True)

#Categorical to numerical data

model_mapping = {'Focus':'1'}
cars['model'] = cars['model'].map(model_mapping).fillna(value='1')

fuel_mapping = {'Petrol':'1', 'Diesel':'2'}
cars['fuel type'] = cars['fuel type'].map(fuel_mapping)

transmission_mapping = {'Manual':'1', 'Automatic':'2', 'Semi-Auto':'3'}
cars['transmission'] = cars['transmission'].map(transmission_mapping)

#Fill NaN values in year column

train = cars[cars['year'].notna()]
test = cars[cars['year'].isna()]
print(test.count())
X_train = train.drop(['year'], axis=1)
Y_train = train['year']
X_test = test.drop(['year'], axis=1)

svc = SVC()
svc.fit(X_train, Y_train)
Y_test = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train)*100, 2)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_test = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train)*100, 2)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_test = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

decision_tree = DecisionTreeClassifier(max_depth=1000)
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

models = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN','Random Forest','Decision Tree'],
                'Score': [acc_svc, acc_knn,
              acc_random_forest, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

#Fill NaN values  found with DecisionTrees

ind =cars.index[cars['year'].isna()].tolist()
Y_pred = pd.Series(Y_pred.T, index=ind)
interog = cars['year'].index.isin(ind)
print(Y_pred)

#Robust Scaler

cars_categ = cars.copy()
transformer = RobustScaler().fit(cars_categ)
cars_categ = transformer.transform(cars_categ)

#Heatmap with all features

cars_categ = pd.DataFrame(cars_categ, columns=['model', 'year', 'price', 'transmission', 'engine size', 'mileage', 'fuel type'])
corrmat = cars_categ.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmin=0 ,vmax=1)
cols = corrmat.nlargest(5, 'price')['price'].index
cm = np.corrcoef(cars_categ[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
cars = cars.drop(columns=['fuel type', 'model'])

#Fill NaN part2

cars[interog] = cars[interog].T.fillna(Y_pred).T
fan = cars[interog]
cars['year'] = cars['year'].fillna(fan['year'])

#Model for price

X = cars.drop(['price'], axis=1)
Y = cars['price']
X_train_price, X_test_price, Y_train_price, Y_test_price = train_test_split(X, Y, test_size=0.33, random_state=42)

model_performance = pd.DataFrame(columns = ["Features", "Model", "Score"])
models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), SVR(), RandomForestRegressor(), MLPRegressor()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Linear","Model": model, "Score": score}, ignore_index= True)
print(cars_categ)
print(model_performance)
