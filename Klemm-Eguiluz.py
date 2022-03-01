import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def klemm_eguiluz(n, m, mu):
    # initial condition of complete graphs with m nodes
    G = nx.complete_graph(m)
    # used to represent whether or not a node is activated(1) or deactivated(0)
    # index of activation list will refer to the individual nodes
    activation = list(np.ones(m))
    mu_factor = np.random.ranf(m)
    
    for i in range(m, n):
        # generate list of edges which are to be randomly rewired to preferential attachment
        mu_factor = np.random.ranf(m) < mu
        activated = np.where(np.array(activation) == 1)[0]

        targets = set([activated[j] for j in range(m) if mu_factor[j] == 0])
        
        # Linear preferential attachment
        p_total = np.array(list(map(G.degree, range(i))))
        p_dstr = p_total / sum(p_total)
        
        while len(targets) < m:
            targets.add(np.random.choice(list(range(i)), p = p_dstr))
        G.add_edges_from(zip(np.full(m, i), list(targets)))
        
        # Activation of new node and deactivation where p = a / k, and 1/a = sum of 1/k
        k =  np.array([G.degree(active)**-1 for active in activated])
        p_deact = k / sum(k)
        deactivated = np.random.choice(activated, p = p_deact)
        activation[deactivated] = 0
        activation.append(1)
                
    return G

def test(mu):
    G = klemm_eguiluz(1000, 20, mu)
    return nx.average_clustering(G)

test = np.vectorize(test)

def mass_test(mu):
    res = test(np.full(100, mu))
    print(f"Done with mu = {mu}")
    return np.mean(res)

mass_test = np.vectorize(mass_test)

results = mass_test(np.arange(0, 1, 0.01))
plt.plot(np.arange(0, 1, 0.01), results)
plt.xlabel("Mu")
plt.ylabel("Clustering Coefficient")
plt.show()