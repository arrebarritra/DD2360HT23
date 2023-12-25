import sys
import matplotlib.pyplot as plt
import pandas as pd
import math

file = "./res_flops.csv"

df = pd.read_csv(file)
ax = df.plot(x='dimX', y='flops', logx=True, legend=None)
ax.set_xticks(df.dimX)
ax.set_xticklabels(["$2^{" + str(int(math.log2(dimX))) + "}$" for dimX in df.dimX])

plt.xlabel("dimX")
plt.ylabel("FLOPS")
plt.savefig("plt_flops.png")