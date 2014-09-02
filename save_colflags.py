import pandas as pd
import numpy as np
import sqlite3 as sqlite
from bkputils import *

import colflags

write("loading data")
conn = sqlite.connect("data.sqlite")
traindata = pd.read_sql("SELECT * FROM training", conn)
feature_cols = [colName for colName in traindata.columns.values if colName.startswith("DER") or colName.startswith("PRI")]
# feature_cols = filter(lambda colName: traindata.dtypes[colName] == np.float64, feature_cols)
writeDone()

write("calcing colflags")
traindata = traindata.applymap(lambda x: np.nan if x == -999.0 else x)
traindata["col_flag"] = colflags.calc_col_flags(traindata, feature_cols)
writeDone()

write("calcing col_flag_strs")
traindata["col_flag_str"] = [
	"{0:b}".format(row["col_flag"])
	for index, row in traindata.iterrows()
]
writeDone()

write("updating sqlite database")
conn.executemany(
	"UPDATE training SET col_flag=?, col_flag_str=? WHERE EventId=?",
	traindata[["col_flag", "col_flag_str", "EventId"]].values
)
conn.commit()
writeDone()