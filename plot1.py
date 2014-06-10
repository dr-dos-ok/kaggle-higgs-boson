import numpy as np
import pylab as P
import sqlite3

conn = sqlite3.connect("data.sqlite")
cursor = conn.cursor()

print "selecting data..."
data = cursor.execute("SELECT DER_mass_MMC FROM test WHERE DER_mass_MMC != -999.0").fetchall()
data = np.array(data)

print "plotting histogram..."
n, bins, patches = P.hist(data, 200, range=(0, 120))

print "showing histogram..."
P.show()
print "Done."