import numpy as np
import pandas as pd
import neuralnet as nn

import bkputils, sys

VELOCITY_DECAY = 0.99
LEARNING_RATE = 0.01
BATCH_SIZE = 1

def z_score(col):
	"""Calculate the z-scores of a column or pandas.Series"""
	mean = col.mean()
	std = col.std()
	return (col - mean)/std

def z_scores(df):
	"""Calculate the z-scores of a dataframe on a column-by-column basis"""
	return df.apply(z_score, 0)

iris = pd.read_csv("iris.csv")

feature_cols = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
zscore_cols = ["z" + col for col in feature_cols]
output_cols = ["is_setosa", "is_versicolor", "is_virginica"]

iris["is_setosa"] = (iris["Name"] == "Iris-setosa").astype(np.float64)
iris["is_versicolor"] = (iris["Name"] == "Iris-versicolor").astype(np.float64)
iris["is_virginica"] = (iris["Name"] == "Iris-virginica").astype(np.float64)

zscores = z_scores(iris[feature_cols])
zscores.columns = zscore_cols
iris[zscore_cols] = zscores

net = nn.FeedForwardNet(
	[len(feature_cols), 5, len(output_cols)],
	err_fn=nn.quadratic_error
)

errs, err_dervs = nn.squared_error(
	net.forward(iris[zscore_cols].values),
	iris[output_cols].values
)
print np.mean(errs)

weights = net.get_flattened_weights()
velocity = net.zeros_like_flattened_weights()

while not bkputils.is_cancelled():
	batch_indices = np.random.choice(iris.shape[0], BATCH_SIZE, replace=False)
	batch = iris.ix[batch_indices]

	avg_grad = net.get_partial_derivs(
		batch[zscore_cols].values,
		batch[output_cols].values,
		outputs=nn.FLATTENED_OUTPUTS
	)
	
	velocity *= VELOCITY_DECAY
	velocity += -avg_grad * LEARNING_RATE
	weights += velocity
	net.set_flattened_weights(weights)
	# print weights

	errs, err_dervs = nn.squared_error(
		net.forward(iris[zscore_cols].values),
		iris[output_cols].values
	)
	msg = "\rerr: %1.9f" % np.mean(errs)
	# print msg
	# __ = raw_input()
	sys.stdout.write(msg)
	sys.stdout.flush()
print

nn.printl("weights", net.weights)
nn.printl("bias_weights", net.bias_weights)

sample = net.forward(iris[zscore_cols].values)
sampledf = pd.DataFrame(sample, columns=["x" + col for col in output_cols])
for col in output_cols:
	sampledf[col] = iris[col]
print
print sampledf[sampledf["is_setosa"]==1.0].iloc[:5].to_string()
print sampledf[sampledf["is_versicolor"]==1.0].iloc[:5].to_string()
print sampledf[sampledf["is_virginica"]==1.0].iloc[:5].to_string()