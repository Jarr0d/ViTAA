import os
import json

def makedir(root):
    if not os.path.exists(root):
        os.makedirs(root)

