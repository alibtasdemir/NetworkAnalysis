import networkx as nx
import numpy as np
import pandas as pd
import os
import glob

graph_folder = os.path.abspath("graphs/")
file_pattern = os.path.join(graph_folder, "*.edges")

feature_categories = np.array(["Name", "Vertex", "Edges", "Density", "Max. Degree", "Avg. Degree", "Max. k-core", "Avg. Clustering Coeff.",
                      "Diameter", "Avg. Path Length", "Total Triangles", "Avg. Eigenvector Centrality"])



def getFeatureVect(G, name="NoName"):
    values = []
    degree_list = np.array(list(G.degree()), dtype=int)[:,1]
    values.append(name)
    values.append(nx.number_of_nodes(G))
    values.append(nx.number_of_edges(G))
    values.append(nx.density(G))
    values.append(np.max(degree_list))
    values.append(np.mean(degree_list))
    values.append(max(list(nx.core_number(G).values())))
    print("Halfway Done!")
    values.append(nx.average_clustering(G))
    values.append(nx.diameter(G))
    print("Almost finished!")
    values.append(nx.average_shortest_path_length(G))
    print("Making final Calculations!")
    values.append(sum(list(nx.triangles(G).values()))/3)
    print("Last one.")
    values.append(np.mean(np.array(list(nx.eigenvector_centrality_numpy(G).values()))))
    return np.array(values)

df = pd.DataFrame(columns=feature_categories)
i = 0

for filename in glob.glob(file_pattern):
    print(filename)
    G = nx.read_edgelist(filename)
    file = os.path.basename(filename)
    G = nx.read_edgelist(filename)
    data = getFeatureVect(G, file)
    df.loc[i] = data
    i+=1
    del(G)
    print("%s --- Extracted"%(file))

with open("features.csv", "w", newline='') as f:
    f.write(df.to_csv())
    f.close()
