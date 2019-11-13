import networkx as nx
import numpy as np
import pandas as pd
import os
import glob
import time

start_time = time.time()
# The graph folder.
graph_folder = os.path.abspath("graphs/")
# File pattern of the graph files.
file_pattern = os.path.join(graph_folder, "*.edges")

# Extracted features.
# To create data frame.
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

if __name__ == "__main__":
	# Empty data frame created
	df = pd.DataFrame(columns=feature_categories)
	# Index counter
	i = 0

	# Scanning for all files with defined file pattern. 
	for filename in glob.glob(file_pattern):
		print(filename)
		file = os.path.basename(filename)
		# Creating graph from edge list
		read_time = time.time()
		G = nx.read_edgelist(filename)
		print("##### %s --- Read completed in %s seconds" %(file, time.time() - read_time))
		# Calculating graph's features
		feature_time = time.time()
		data = getFeatureVect(G, file)
		print("##### %s --- Calculations completed in %s seconds" %(file, time.time() - feature_time))
		# Placing on data frame
		df.loc[i] = data
		i+=1
		# Deleting graph to save memory
		del(G)
		print("%s --- Extracted"%(file))

	# Creating a csv to save features.
	with open("features.csv", "w", newline='') as f:
	    f.write(df.to_csv())
	    f.close()
	
	print("##### Total execution time is %s seconds" % (time.time()-start_time))
