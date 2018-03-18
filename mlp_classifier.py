import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

input_data = pd.read_csv("input_data.csv")
output_data = pd.read_csv("outputs.csv")

print(input_data.shape)
print(output_data.shape)

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, random_state=0)

# print(y_train.ix[:,0])


mlp = MLPClassifier(activation="logistic", solver="sgd", random_state=1, hidden_layer_sizes=(10, 15))
mlp.fit(x_train, y_train.ix[:,0])

print("train successful")
print("Accuracy on train data:", mlp.score(x_train, y_train.ix[:,0]))
print("Accuracy on test data:", mlp.score(x_test, y_test))


# for index, row in pred.iterrows():
#     print(row)