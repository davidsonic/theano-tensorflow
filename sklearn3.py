from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data=datasets.load_boston()
data_X=loaded_data.data
data_y=loaded_data.target

model=LinearRegression()
model.fit(data_X,data_y)

print(model.coef_)
print(model.intercept_)

print(model.get_params())  #model default parameters

print(model.score(data_X,data_y))  #score the model R^2 coefficient of determination