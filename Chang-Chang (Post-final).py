import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import Counter

def chang_chang(n, k, phi, chi, delta):
    
    g = nx.Graph()
    
    def post_gen(irrelevant):
        theta, norm = np.random.ranf() * np.pi * 2, np.random.ranf()
        x, y = np.cos(theta) * norm, np.sin(theta) * norm
        return [x, y]
    
    clusters = [[int(i)] for i in range(chi)]
    cluster_b = [tuple(post_gen(1)) for i in clusters]
    posts = cluster_b.copy()
    
    def cluster_b_gen(node):
        return (node, dict(pos = cluster_b[node]))
    
    initial = list(map(cluster_b_gen, range(chi)))
    g.add_nodes_from(initial)
    
    def edge_initialize(node):
        target = np.random.choice(range(chi))
        return (node, target)
    
    edges_initialize = list(map(edge_initialize, range(chi)))
    g.add_edges_from(edges_initialize)
    
    for node in range(chi, n):
        pplns = list(map(len, clusters))
        ppln = list(np.array(pplns) / sum(pplns))
        cluster_add = np.random.choice(range(chi), p = ppln)
        i = cluster_add
        clusters[i].append(node)
        
        def delt_gen(node):
            delt = post_gen(1)
            delt = tuple([delt[0] / delta, delt[1] / delta])
            post = (cluster_b[i][0] + delt[0], cluster_b[i][1] + delt[1])
            posts.append(post)
            return post
        
        g.add_node(node, pos = delt_gen(node))

        def edges(k):
            
            def distance(target):
                dist = ((posts[node][0] - posts[target][0]) ** 2 + (posts[node][1] - posts[target][1]) ** 2)**0.5
                return dist
            
            def pcalc(target):
                d = distance(target)
                return ((g.degree(target)+1)**phi) / d
            
            def p_dist(targets):
                dist = np.array(list(map(pcalc, targets)))
                dist = dist / sum(dist)
                return list(dist)
            
            size_n = np.random.randint(1, k)
                
            available = list(range(0, node))
            targets = np.random.choice(available, p = p_dist(available), size = size_n, replace = False)
            edges_add = list(zip(targets, np.full(len(targets), node)))
        
            return edges_add
    
        g.add_edges_from(edges(k))
    
    print(f"\nNo. of clusters: {len(clusters)}")
    return g


toc = time.time()
#   chang_chang(n, k, phi, chi, delta)
g = chang_chang(300, 20, 2, 20, 3)
tic = time.time()
print(nx.info(g))
print(f"\nGlobal clustering coefficient: {nx.average_clustering(g)}")
print(f"Transitivity: {nx.transitivity(g)}")
print(f"Time taken: {tic-toc}")
pos = nx.get_node_attributes(g, 'pos')
nx.draw(g, pos, node_size = 50)
plt.show()
plt.clf()

degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
count = Counter(degree_sequence)

plt.bar(count.keys(), count.values(), color='b')
plt.title("Degree Histogram")
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.show()


'''
Typically, keep value of delta around 5 or less, and phi around 5 or less as well.
Chi should be adjusted to fit n, same with k. Chi must be more than k, to avoid instances of 
k edges added being less than chi nodes in the beginning.

Clustering coefficient is pretty high, but transitivity is damn low.
Edges are randomly connected, and can see that is is connected all across the graph, 
not similar to Facebook
Degree distribution is scale free, but there appears to be gaps at low degrees, 
where there are no nodes with that specific degree sometimes though the degrees close to it can have
more than a quarter of the nodes

Also note that the cluster base is determined using randomised polar coordinates, and now so is
the scatter around the base.

'''

