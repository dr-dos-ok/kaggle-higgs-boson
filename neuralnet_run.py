import numpy as np
import pandas as pd
import neuralnet as nn

iris = pd.read_csv("iris.csv")

print pd.Series(iris["Name"].values).unique()

iris["is_setosa"] = iris["Name"] == "Iris-setosa"
iris["is_versicolor"] = iris["Name"] == "Iris-versicolor"
iris["is_virginica"] = iris["Name"] == "Iris-virginica"

def train_backprop_minibatch(net, batch, learning_rate=2.0):
	avg_grad = net.zeros_like_flattened_weights()
	for row in batch.iterrows():
		avg_grad += net.get_partial_derivs(row[feature_cols], row[output_cols])
	avg_grad = nn.normalize_vector(avg_grad)
	weights = net.get_flattened_weights()
	weights += (-avg_grad) * learning_rate
	net.set_flattened_weights(weights)

feature_cols = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
output_cols = ["is_setosa", "is_versicolor", "is_virginica"]

iris["is_setosa"] = iris["Name"] == "Iris-setosa"
iris["is_versicolor"] = iris["Name"] == "Iris-versicolor"
iris["is_virginica"] = iris["Name"] == "Iris-virginica"

net = nn.FeedForwardNet([len(feature_cols), 10, 3])

