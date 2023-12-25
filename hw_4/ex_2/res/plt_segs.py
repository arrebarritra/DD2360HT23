import sys
import matplotlib.pyplot as plt
import pandas as pd

file = "./res_segs.csv"
df = pd.read_csv(file)
ax = df.plot(x='segs', y='mean', logx=True, legend=None)
ax.set_xticks(df.segs)
ax.set_xticklabels(df.segs)

plt.xlabel("N segments")
plt.ylabel("Execution time (ms)")
plt.savefig("plt_segs.png")