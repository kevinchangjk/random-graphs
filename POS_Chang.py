import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import time
from scipy.stats import linregress

def pos_chang(function, best, start, end):
    def chi_round(x):
        return int(round(x, 0))
    
    start = np.array(start)
    end = np.array(end)
    n = 10
    phi_init = np.arange(start[0], end[0], (end[0]-start[0]) / n)
    chi_init = np.array(list(map(chi_round, np.arange(start[1], end[1], (end[1]-start[1]) / n))))
    print(chi_init)
    delta_init = np.arange(start[2], end[2], (end[2]-start[2]) / n)
    n_post = np.array([[phi_init[i], chi_init[i], delta_init[i]] for i in range(n)])
    gbest = [0, np.zeros(3)]
    pbest_q = [0 for i in range(n)]
    pbest_p = n_post.copy()
    drct = np.random.ranf([n, 3]) * np.array([[end-start] for i in range(n)] * (-1)**np.random.choice([1, 2], size = 3 * n).reshape(n, 3))
    
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
        dirac = 0.25 * drct[i] + 0.5 * np.random.random() * (pbest_p[i] - n_post[i]) + np.random.random() * (gbest[1] - n_post[i])
        return dirac
    
    swarm = np.vectorize(swarm)
    update = np.vectorize(update)
    update(range(n))
    
    while np.std(pbest_q) > 10**(-5):
        drct = np.array(list(map(swarm, range(n))))
        n_post = n_post + drct
        
        phi_post = np.clip(n_post[:, 0], start[0], end[0])
        chi_post = np.clip(np.array(list(map(chi_round, n_post[:, 1]))), start[1], end[1])
        delta_post = np.clip(n_post[:, 2], start[2], end[2])
        n_post = np.array([[phi_post[i], chi_post[i], delta_post[i]] for i in range(n)])
        update(range(n))
        
    return [np.array([np.mean([pbest_p[i, j] for i in range(n)]) for j in range(3)]), np.mean(pbest_q)]

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

n = 100
k = 10

def test_chang(post):
    phi, chi, delta = post
    g = chang_chang(n, k, phi, chi, delta)
    cluster_score = np.mean(nx.average_clustering(g) + nx.transitivity(g))
    distance_score = 1 / nx.average_shortest_path_length(g)
    degree_sequence = sorted([d for n, d in g.degree()])
    count = Counter(degree_sequence)
    x = np.array(count.keys())
    y = np.array(count.values())
    dd_score = linregress(x, y)[2] ** 2
    return np.mean([cluster_score, distance_score, dd_score])

print(pos_chang(test_chang, 1, [0.5, k, 0.5], [10, n, 10]))
