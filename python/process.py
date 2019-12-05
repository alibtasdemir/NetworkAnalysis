"""
This code provides automation for ESCAPE algorithm (https://bitbucket.org/seshadhri/escape/src/master/) if
large amount of graph files will be proccessed.
"""

import glob
import os
import numpy as np
import pandas as pd

# PRE-DEFINED OS VARIABLES
py2 = "python"
py3 = "python3"
SPACE = " "

# Paths of folders and files.
sanitizer = os.path.abspath("../python/sanitize.py")
graph_folder = os.path.abspath("../graphs/")
to_sanitizer_pattern = os.path.join(graph_folder, "*.txt")
edge_list_pattern = os.path.join(graph_folder, "*.edges")

print("Sanitizing txt files")

# Sanitizing is handling which required step for ESCAPE algorithm.
for filename in glob.glob(to_sanitizer_pattern):
    # python sanitize.py <FOLDER> <FILE>
    command = py2 + SPACE + sanitizer + SPACE + graph_folder + SPACE + filename
    os.system(command)

print("Reading Edge Lists")

# Creating pandas data frame to store feature lists
feature_categories = ["f"+str(i) for i in range(1,35)]
feature_categories.insert(0, "Name")
df = pd.DataFrame(columns=feature_categories)
# Data frame index counter
i = 0

for filename in glob.glob(edge_list_pattern):
    print("#"*10 + SPACE + filename + SPACE + "#"*10)
    # python3 subgraph_counts.py <FILE> <OPTIONS> <K>
    command = py3 + SPACE + "subgraph_counts.py "+ filename + " 5"
    os.system(command)
    # Reading output.
    with open("out.txt", "r") as f:
        stream = f.readlines()
        features = [i.strip() for i in stream]
        # We take only 5clique motifs
        features = features[17:]
        # Adding filename to create data.
        features.insert(0, os.path.basename(filename))
        features = np.array(features)
        f.close()
    # Placing the feature vector into Data frame and update the counter.
    df.loc[i] = features
    i+=1

# Creating csv file with data frame built by features of graphs.
with open("features.csv", "w", newline='') as f:
    f.write(df.to_csv())
    f.close()
