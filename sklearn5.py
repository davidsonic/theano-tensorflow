'''
cross validation 1
'''


import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()
iris_X=iris.data
iris_y=iris.target


X_train,X_test,y_train,y_test=train_test_split(
    iris_X,iris_y,random_state=4
)


from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

# eg.1
knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,iris_X,iris_y,cv=5,scoring='accuracy')  #  5 groups

print scores
print(scores.mean())

#eg.2 find parameters
k_range=range(1,31)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=-cross_val_score(knn,iris_X,iris_y,cv=10,scoring='mean_squared_error') #for regression
    # scores=cross_val_score(knn,iris_X,iris_y,cv=10,scoring='accuracy') #for classification
    k_scores.append(scores.mean())

plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validation Accuracy')
plt.show()

