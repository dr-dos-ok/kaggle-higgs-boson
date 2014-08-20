import numpy as np
import pandas as pd
import neuralnet as nn

iris = pd.read_csv("iris.csv")

print iris.head()

net = nn.FeedForwardNet([iris.shape[1], 100])