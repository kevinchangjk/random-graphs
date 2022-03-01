import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def barabasi_albert(n, m):
    g = nx.Graph()
    for initial in range(m):
        g.add_node(initial)
    for i in range(n-m):
        g.add_node(i+m)
        if i == 0:
            for node in range(i+m):
                g.add_edge(node, i+m)
        else:
            edges = 0
            while edges < m:
                node_preferential = []
                for node in range(i+m):
                    for repeat in range(g.degree(node)):
                        node_preferential.append(node)
                node_chosen = np.random.choice(node_preferential)
                if g.has_edge(node_chosen, i+m) == True:
                    continue
                else:
                    g.add_edge(node_chosen, i+m)
                    edges+=1
    return g

graph = barabasi_albert(10, 3)
print(nx.info(graph))
nx.draw(graph)
plt.show()
plt.clf()

hraph = nx.barabasi_albert_graph(10, 3)
print(nx.info(hraph))
nx.draw(hraph)
plt.show()