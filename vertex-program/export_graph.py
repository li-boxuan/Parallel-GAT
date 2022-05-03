"""
This utility script converts PPI graph data from adjacency list format to
TinkerPop Kryo format.
"""

from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

if __name__ == "__main__":
    g = traversal().withRemote(DriverRemoteConnection('ws://localhost:8182/gremlin','g'))
    # clear existing graph
    g.V().drop().iterate()

    f = open("../data/ppi_adj.txt", "r")
    f2 = open("../data/ppi_features.txt", "r")
    f3 = open("../data/ppi_labels.txt", "r")
    lines = f.readlines()
    features_list = f2.readlines()
    labels_list = f3.readlines()
    nodes = int(lines[0].split()[0])
    nodeMap = dict()
    # create vertices
    for i in range(nodes):
        node = g.addV().property("vid", i).next()
        nodeMap[i] = node
        features = features_list[i].split()
        for j in range(len(features)):
            g.V(node).property("feature" + str(j), float(features[j])).next()
        labels = labels_list[i].split()
        for j in range(len(labels)):
            g.V(node).property("label" + str(j), float(labels[j])).next()
    f2.close()
    f3.close()

    # create edges
    col_indices = lines[1].split()
    delim = lines[2].split() 
    for i in range(nodes):
        v1 = nodeMap[i]
        start_idx = int(delim[i])
        end_idx = int(delim[i+1])
        for k in range(start_idx, end_idx):
            neighbor_idx = int(col_indices[k])
            v2 = nodeMap[neighbor_idx]
            g.V(v1).addE("connect").to(v2).next()
    f.close()

    output_path = "/Users/liboxuan/open-source/tinkerpop/gremlin-test/src/main/resources/org/apache/tinkerpop/gremlin/structure/io/gryo/ppi-v3d0.kryo"
    g.io(output_path).write().iterate()