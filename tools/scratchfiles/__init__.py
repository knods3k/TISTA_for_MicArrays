import sys
import os

# Add paths to source files relative to this file so they can be used in Import statements under test/
sys.path.append(os.path.join(sys.path[0], "tools/"))
sys.path.append(os.path.join(sys.path[0], "../"))