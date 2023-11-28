import sys
import matplotlib.pyplot as plt

if(len(sys.argv) > 1):
    file = sys.argv[1]
else:
    print("Provide file in argument")
    sys.exit()

counts = []
with open(file) as f:
    for line in f:
        counts.append(int(line))

plt.bar(list(range(len(counts))), counts)
plt.xlabel("bin")
plt.ylabel("count")
plt.show()