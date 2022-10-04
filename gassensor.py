from xml.etree.ElementTree import tostring
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold

gassensor_CO = pd.read_csv("gas-sensor-data_04042022.csv")
# gassensor_CO2 = pd.read_csv("gas-sensor-data_04042022.csv")

y = gassensor_CO["AQMesh-CO"]
# y2 = gassensor["AQMesh-CO2"]
gassensor_CO.drop(["Time","AQMesh-CO","AQMesh-CO2","CO2 IRC Voltage"], axis=1, inplace=True)
# gassensor_CO2.drop(["Time","AQMesh-CO","AQMesh-CO2","CO Auxiliary Voltage"], axis=1, inplace=True)

num_of_obs = len(y)

###################################

# Code for test tree parameters

X_train, X_test, y_train, y_test = train_test_split(gassensor_CO, y, test_size = 0.1)

def get_tree_models():
    models = dict()
    n_trees = [10,50,100,500,1000]
    for n in n_trees:
        models[str(n)] = RandomForestRegressor(n_estimators=n)
    return models

def get_features_models():
    models = dict()
    for n in range(1,7):
        models[str(n)] = RandomForestRegressor(max_features=n)
    return models

def get_tree_depth_models():
    models = dict()
    depth = [i for i in range(1, 8)] + [None]
    for n in depth:
        models[str(n)] = RandomForestRegressor(max_depth=n)
    return models

def get_tree_randomness_models():
    models = dict()
    depth = [i for i in range(0, 5)] + [None]
    for n in depth:
        models[str(n)] = RandomForestRegressor(random_state=n)
    return models

def evaluate_model(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ypredit = model.predict(xtest)
    errors = abs(ypredit - ytest)
    mape = 100 * (errors / ytest)
    accuracy = 100 - np.mean(mape)
    return accuracy

names, scores = list(), list()

# TODO change the function to test for different parameter
models = get_tree_randomness_models()

for name, model in models.items():
    for n in range(1,10):
        scores.append(evaluate_model(model, X_train, y_train, X_test, y_test))
    names.append(name)
    print('>%s %.3f %.3f' % (name, np.mean(scores), np.std(scores)))

###################################

# Test output 

# for number of tress, 1000 will be choose
# >10 96.465 0.046
# >50 96.523 0.071
# >100 96.558 0.077
# >500 96.577 0.075
# >1000 96.590 0.072

# for number of features, 2 will be choose
# >1 96.404 0.042
# >2 96.446 0.055
# >3 96.444 0.048
# >4 96.444 0.046
# >5 96.441 0.043

# for depth of tree, the performance improve for increase in tree depth, so the default, no maximum depth will be choose
# >1 92.575 0.033
# >2 93.568 0.993
# >3 94.090 1.097
# >4 94.445 1.132
# >5 94.695 1.129
# >6 94.886 1.115
# >7 95.040 1.100
# >None 95.190 1.102

# For randomness of the model, 0 is the most favourd. 
# >0 96.125 0.000
# >1 96.079 0.046
# >2 96.074 0.038
# >3 96.071 0.034
# >4 96.066 0.032
# >None 96.070 0.033

###################################

# Code for test test set size 

# def evaluate_model_separate(n):
#     model = RandomForestRegressor()
#     X_train, X_test, y_train, y_test = train_test_split(gassensor_CO, y, test_size = n)
#     model.fit(X_train, y_train)
#     ypredit = model.predict(X_test)
#     errors = abs(ypredit - y_test)
#     mape = 100 * (errors / y_test)
#     accuracy = 100 - np.mean(mape)
#     return accuracy

# separation = [0.1, 0.2, 0.3, 0.4]
# for n in separation:
#     scores.append(evaluate_model_separate(n))
#     print('>%s %.3f %.3f' % (n, np.mean(scores), np.std(scores)))

###################################

# Test output

# Although 0.1 do not give the best prediction. Consider that it has the lowest std, 0.1 will be chosen
# >0.1 96.357 0.000
# >0.2 96.509 0.152
# >0.3 96.318 0.298
# >0.4 96.277 0.268

###################################

# # TODO Given the optimal test set size, the n_splits is derived from dividing 1 by the optimal test size
# n_test_size = 1/0.1
# kf = RepeatedKFold(n_splits = n_test_size, n_repeats = 10, random_state = None) 

# # TODO please change the parameters accordingly from the previous testing 
# regressor = RandomForestRegressor(n_estimators = 1000, max_features= 2)

# # Count the folds, do not change dummy variable
# num_of_fold = 1

# for train_index, test_index in kf.split(gassensor_CO):
#     # Get the test and train set 
#     X_train, X_test = gassensor_CO.loc[train_index,], gassensor_CO.loc[test_index,] 
#     y_train, y_test = y[train_index], y[test_index]

#     # Predict the result 
#     regressor.fit(X_train, y_train)
#     y_pred_test = regressor.predict(X_test)

#     print("Number of fold: ", num_of_fold)

#     # Calculate the absolute errors
#     errors = abs(y_pred_test - y_test)
#     print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#     # Calculate mean absolute percentage error (MAPE)
#     mape = 100 * (errors / y_test)

#     # Calculate and display accuracy
#     accuracy = 100 - np.mean(mape)
#     print('Accuracy:', round(accuracy, 2), '%.')

#     # Calculate adjusted R square
#     corr_matrix = np.corrcoef(y_test, y_pred_test)
#     corr = corr_matrix[0,1]
#     R_sq = corr ** 2

#     # TODO please change the number of predictors accordingly - number of features
#     num_of_predictor = 2
#     adj_R2 = 1 - ((1 - R_sq) * (num_of_obs - 1)/(num_of_obs - num_of_predictor - 1))
#     print("Adjusted R square is: ", adj_R2)

#     num_of_fold += 1
#     print("")

#     if num_of_fold == 11:
#         break