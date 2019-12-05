"""
Searches a directory and prints out not connected graphs in that directory.
"""
import networkx as nx
import os
import glob

# Directory path
graph_folder = os.path.abspath("Random_soc-firm-hi-tech/")
# File extensions to be searched.
file_pattern = os.path.join(graph_folder, "*.txt")

for filename in glob.glob(file_pattern):
    #print(filename)
    G = nx.read_edgelist(filename)
    if not nx.is_connected(G):
        print(filename)
