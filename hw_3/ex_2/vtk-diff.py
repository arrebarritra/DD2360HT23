import sys
import os
import numpy as np

def loadVTK(path):
    vals = []
    with open(path) as f:
        for _ in range(10):
            next(f)
        
        for line in f:
            strvals = str.split(line)
            for strval in strvals:
                vals.append(float(strval))
    return np.array(vals)


if(len(sys.argv) > 2):
    folder1 = sys.argv[1]
    folder2 = sys.argv[2]
else:
    print("Provide 2 folders in argument")
    sys.exit()

files1 = [f for f in os.listdir(folder1) if f[-4:]==".vtk"]
files2 = [f for f in os.listdir(folder2) if f[-4:]==".vtk"]

for file in files1:
    if file in files2:
        dat1 = loadVTK(os.path.join(folder1, file))
        dat2 = loadVTK(os.path.join(folder2, file))

        assert(dat1.size == dat2.size)
        print(f"{file}: loaded {dat1.size} values")
        max_diff = np.max(np.abs(dat1 - dat2))
        print(f"Max discrepancy: {max_diff}\n")