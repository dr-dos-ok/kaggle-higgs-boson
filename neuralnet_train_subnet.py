import pandas as pd
import numpy as np
import neuralnet as nn

from bkputils import *
from neuralnet_train import train
from neuralnet_classifier import ZNetClassifier

import colflags, sys

available_colflags = [
	"111111111111111111111111111111",
	"111111111111111111111111111110",
	"100011111111111110111110001111",
	"100011111111111110111110001110",
	"100000011111111110111110001111",
	"100000011111111110111110001110"
]

if len(sys.argv) < 2:
	print "Please pass an index between 1 and 6 as a command line parameter. E.g.: \"python neuralnet_train_subnet.py 6\""
	exit()
col_flag_index = int(sys.argv[1]) - 1
col_flag_str = available_colflags[col_flag_index]
col_flag = int(col_flag_str, 2)

write("training subnet for {0:b}".format(col_flag))

write("loading training data")
traindata = loadTrainingData(col_flag_str=col_flag_str)
feature_cols = featureCols(only_float64=False)
feature_cols = colflags.get_flagged_cols(col_flag, feature_cols)
traindata["train_feature"] = np.ones(traindata.shape[0])
traindata.loc[traindata["Label"]=="b", "train_feature"] = -1
writeDone()

write("training net")
net = nn.ZFeedForwardNet(traindata, feature_cols, [-1, 1000, 1])
train(net, traindata)
writeDone()

write("classifying traindata")
classifier = ZNetClassifier(net)
predictions, confidence = classifier.classify(traindata)
score = ams(predictions, traindata)
writeDone()
print "\t\tams: {0}".format(score)

write("saving net")
net.save("{0}.nn".format(col_flag_str))
writeDone()

writeDone()