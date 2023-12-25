import sys
import matplotlib.pyplot as plt
import pandas as pd
import math

file = "./res_steps.csv"

df = pd.read_csv(file)
ax = df.plot(x='steps', y='rerror', logx=True, legend=None)
ax.set_xticks(df.steps)
ax.set_xticklabels(["$2^{" + str(int(math.log2(steps))) + "}$" for steps in df.steps])

plt.xlabel("Steps")
plt.ylabel("Relative error")
plt.savefig("plt_steps_error.png")