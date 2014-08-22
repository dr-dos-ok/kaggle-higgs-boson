import numpy as np
import pandas as pd
import neuralnet as nn

import signal

ITERATIONS = 1000

def z_score(col):
	"""Calculate the z-scores of a column or pandas.Series"""
	mean = col.mean()
	std = col.std()
	return (col - mean)/std

def z_scores(df):
	"""Calculate the z-scores of a dataframe on a column-by-column basis"""
	return df.apply(z_score, 0)

sigint_passed = False
def handle_sigint(signal, frame):
	global sigint_passed
	sigint_passed = True
signal.signal(signal.SIGINT, handle_sigint)

def make_lerp(p1, p2):
	m = (p2[1] - p1[1]) / (p2[0] - p1[0])
	b = p1[1] - (m * p1[0])
	def anon_lerp(x):
		return (m*x) + b
	return anon_lerp

iris = pd.read_csv("iris.csv")

def describe(df):
	print df["Name"].iloc[0]
	print pd.DataFrame({
		"mean": df.mean(),
		"std": df.std()
	}).transpose()
	print

# print iris[iris["Name"]=="Iris-setosa"].describe(percentiles=[])
describe(iris[iris["Name"]=="Iris-setosa"])
describe(iris[iris["Name"]=="Iris-versicolor"])
describe(iris[iris["Name"]=="Iris-virginica"])
exit()

feature_cols = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
zscore_cols = ["z" + col for col in feature_cols]
output_cols = ["is_setosa", "is_versicolor", "is_virginica"]

iris["is_setosa"] = (iris["Name"] == "Iris-setosa").astype(np.int8)
iris["is_versicolor"] = (iris["Name"] == "Iris-versicolor").astype(np.int8)
iris["is_virginica"] = (iris["Name"] == "Iris-virginica").astype(np.int8)

# iris["is_setosa"] = np.ones(iris.shape[0])
# iris["is_versicolor"] = np.zeros(iris.shape[0])
# iris["is_virginica"] = np.zeros(iris.shape[0])

zscores = z_scores(iris[feature_cols])
zscores.columns = zscore_cols
iris[zscore_cols] = zscores
# print iris.ix[:20]
# exit()

net = nn.FeedForwardNet(
	[len(feature_cols), 25, len(output_cols)],
	err_fn=nn.quadratic_error
)

errs, err_dervs = nn.squared_error(
	net.forward(iris[zscore_cols].values),
	iris[output_cols].values
)
print np.mean(errs)


velocity = net.zeros_like_flattened_weights()
learning_rate = 0.01
velocity_decay = 0.9
batch_size = 40 # iris.shape[0]

flattened_weights = net.get_flattened_weights()

nn.printl("weights", net.weights)
nn.printl("bias_weights", net.bias_weights)
print

learning_rate_lerp = make_lerp((0, 0.1), (ITERATIONS-1, 0.1))
velocity_decay_lerp = make_lerp((0, 0.9), (ITERATIONS-1, 0.5))

for i in xrange(ITERATIONS):
	
	inverse_i = float(ITERATIONS - i) / ITERATIONS

	batch_indices = np.random.choice(iris.shape[0], batch_size, replace=False)
	batch = iris.ix[batch_indices]

	avg_grad = net.get_partial_derivs(
		batch[zscore_cols].values,
		batch[output_cols].values,
		outputs=nn.FLATTENED_OUTPUTS
	)
	avg_grad = nn.normalize_vector(avg_grad)

	velocity_decay = velocity_decay_lerp(i)
	learning_rate = learning_rate_lerp(i)

	velocity += (-avg_grad * learning_rate)
	velocity *= velocity_decay

	flattened_weights += velocity
	net.set_flattened_weights(flattened_weights)

	errs, err_dervs = nn.squared_error(
		net.forward(iris[zscore_cols].values),
		iris[output_cols].values
	)
	print np.mean(errs)

	if sigint_passed:
		break

nn.printl("weights", net.weights)
nn.printl("bias_weights", net.bias_weights)

sample = net.forward(iris[zscore_cols].values)
sampledf = pd.DataFrame(sample, columns=["x" + col for col in output_cols])
for col in output_cols:
	sampledf[col] = iris[col]
print
print sampledf.ix[:20]