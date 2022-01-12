import sys
import os
from matplotlib import pyplot as plt

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('figure', figsize=(16,9))

# Add paths to source files relative to this file so they can be used in Import statements under test/
sys.path.append(os.path.join(sys.path[0], "tools/"))
sys.path.append(os.path.join(sys.path[0], "../"))