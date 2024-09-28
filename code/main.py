import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.neural_network

data=pd.read_csv("project_train.csv")
features=data.drop("Label", axis=1)
targets=data["Label"]
X_train, X_test, Y_train, Y_test=train_test_split(features, targets, test_size=0.3, random_state=20)

neural_network=sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(10,10), activation=('relu'))
neural_network.fit(X_train, Y_train)

y_pred=neural_network.predict(X_test)
Y_test.reset_index(drop=True)
s=0

for k in range(0,len(Y_test)):
    if y_pred[k]==Y_test.iloc[k]:
        s=s+1
print(s/len(Y_test))