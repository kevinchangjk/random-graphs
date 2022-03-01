import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import Counter
'''
def static_scale(n, a, m):
    g = nx.empty_graph(n)
    
    while len(g.edges) < m * n:
        p_tot = sum(map(lambda x: (x+1)**(-a), range(n)))
        nodes = np.random.choice(range(n), size = 2, p = [(i+1)**(-a)/p_tot for i in range(n)])
        if g.has_edge(nodes[0], nodes[1]):
            continue
        else:
            g.add_edge(nodes[0], nodes[1])
    
    return g

toc = time.time()
g = static_scale(200, 1, 10)
tic = time.time()
print(nx.info(g))
print(f"\nGlobal clustering coefficient: {nx.average_clustering(g)}")
print(f"Transitivity: {nx.transitivity(g)}")
print(f"Time taken: {tic-toc}")
nx.draw(g)
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

def static_block(n, a, m, chi, zeta, xi):
    groups = []
    nodes = 0
    available = list(range(n))
    
    while nodes < n:
        ppln = np.random.randint(1, n * chi)
        if ppln + nodes > n:
            ppln = n - nodes
        targets = list(set(np.random.choice(available, size = ppln)))
        groups.append(targets)
        nodes += len(targets)
        for target in targets:
            available.remove(target)
    
    '''
    while node < n:
        ppln = np.random.randint(1, n * chi)
        if ppln + node > n:
            ppln = n - node
        cluster = list(range(node, ppln+node))
        groups.append(cluster)
        node += ppln
    '''
    g = nx.empty_graph(n)
    
    while len(g.edges) < m*n:
        p_tot = sum(map(lambda x: (x+1)**(-a), range(n)))
        node = np.random.choice(range(n), p = [(i+1)**(-a)/p_tot for i in range(n)])
        clustered = [cluster for cluster in groups if node in cluster][0]
        
        def p_calc(target):
            if target in clustered:
                return zeta / ((target+1) **(a))
            return xi / ((target+1) ** (a))
        
        p_calc = np.vectorize(p_calc)
        p_dist = p_calc(range(n))
        p_dist = p_dist / sum(p_dist)
        
        target = np.random.choice(range(n), p = p_dist)
        if g.has_edge(target, node):
            continue
        else:
            g.add_edge(node, target)
            
    return g

toc = time.time()
g = static_block(300, 1, 10, 0.2, 0.8, 0.01)
tic = time.time()
print(nx.info(g))
print(f"\nGlobal clustering coefficient: {nx.average_clustering(g)}")
print(f"Transitivity: {nx.transitivity(g)}")
print(f"Time taken: {tic-toc}")
nx.draw(g)
plt.show()
plt.clf()

degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
count = Counter(degree_sequence)

plt.bar(count.keys(), count.values(), color='b')
plt.title("Degree Histogram")
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.show()
