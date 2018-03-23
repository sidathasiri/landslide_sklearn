import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

input_data = pd.read_csv("input_data.csv")
output_data = pd.read_csv("outputs.csv")

#divide train test sets
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, random_state=0)

mlp = MLPClassifier(activation="relu", random_state=1, solver="adam", hidden_layer_sizes=(10, 12, 10))

print("Started training model...")
mlp.fit(x_train, y_train.ix[:, 0])
print("Train complete!")

pred = mlp.predict(x_test)

print("Predicted landslides cells:", (pred == 1).sum())
print("Actual landslides cells:", np.count_nonzero(y_test))

predicitonLoc = np.where(pred == 1)[0]  #predicted landslide locations
actualLoc = np.where(y_test == 1)[0]    #actual landslide locations

#counting correct predictions
counter = 0
for i in predicitonLoc:
    if(i in actualLoc):
        counter += 1

print("Correct predictions: ", counter)
