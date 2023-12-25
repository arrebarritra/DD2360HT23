import sys
import matplotlib.pyplot as plt
import pandas as pd
import math

file = "./res_prefetch.csv"

df = pd.read_csv(file)
ax = df.plot(x='dimX', y=['time_prefetch', 'time_noprefetch'], loglog=True)
ax.set_xticks(df.dimX)
ax.set_xticklabels(["$2^{" + str(int(math.log2(dimX))) + "}$" for dimX in df.dimX])
ax.set_xticklabels(["$2^{" + str(int(math.log2(dimX))) + "}$" for dimX in df.dimX])

plt.legend(['Prefetch', 'No prefetch'])
plt.xlabel("dimX")
plt.ylabel("Time (ms)")
plt.savefig("plt_prefetch.png")