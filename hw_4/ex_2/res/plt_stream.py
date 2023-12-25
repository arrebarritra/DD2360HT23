import sys
import matplotlib.pyplot as plt
import pandas as pd
import math

file = "./res_stream.csv"

df = pd.read_csv(file)
ax = df.plot(x='N', y=df.columns.values[1:], logx=True)
ax.set_xticks(df.N)
ax.set_xticklabels(["$2^{" + str(int(math.log2(N))) + "}$" for N in df.N])

plt.legend([str.split(nseg, "_")[-1] + " segs" for nseg in df.columns.values[1:]])
plt.xlabel("N")
plt.ylabel("Execution time (ms)")
plt.savefig("plt_stream.png")