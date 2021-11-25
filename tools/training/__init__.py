import sys
import os

# Add paths to source files relative to this file so they can be used in Import statements under test/
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)