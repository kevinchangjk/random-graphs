import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import Counter

def pos(function, best, start, end):
    
    n = 10
    n_post = np.arange(start, end, (end-start) / n)
    gbest = [0, np.mean([start, end])]
    pbest_q = [0 for i in range(n)]
    pbest_p = n_post.copy()
    drct = np.random.ranf(n) * (end-start) * (-1)**np.random.choice([1, 2])
    
    def best_calc(post):
        return 1 - (abs(function(post) - best) / best)
    
    def update(i):
        value = best_calc(n_post[i])
        
        if value > gbest[0]:
            gbest[0] = value
            gbest[1] = n_post[i]
        
        elif value > pbest_q[i]:
            pbest_q[i] = value
            pbest_p[i] = n_post[i]
        
    def swarm(i):
        x = 0.25 * drct[i] + 0.5 * np.random.random() * (pbest_p[i] - n_post[i]) + np.random.random() * (gbest[1] - n_post[i])
        return x
    
    swarm = np.vectorize(swarm)
    update = np.vectorize(update)
    update(range(n))
    
    while np.std(pbest_p) > 10**(-5):
        drct = swarm(range(n))
        n_post = np.clip(n_post + drct, start, end)
        update(range(n))
        
    return [np.mean(pbest_p), np.mean(pbest_q)]
    
def ba_power(e):
    n = 100
    m = 10
    g = nx.empty_graph(m)
    g.add_edges_from([(i, m) for i in range(m)])
    
    for i in range(m+1, n):
        g.add_node(i)
        nodes = []
        
        def p_dist(node):
            return g.degree(node)**e
        
        p_dist = np.vectorize(p_dist)
        p = p_dist(list(range(i)))
        p_d = p / sum(p)
    
        nodes = np.random.choice(list(range(i)), size = 4, replace = False, p = p_d)
        g.add_edges_from([(i, node) for node in nodes])
        
    return nx.transitivity(g)

def test(x):
    return np.sin(x) * 5
    
    
toc = time.time()
print(pos(test, 4, 0, 2))
tic = time.time()
print(f"Time taken: {tic-toc}")
    

    