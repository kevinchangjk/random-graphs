import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

def sir(g, si, ir, rs, time):
    print("\nInfection has begun, with a single randomly chosen node being infected.")
    print(f"Chance of susceptible node becoming infected: {si}")
    print(f"Chance of infected node recovering: {ir}")
    print(f"Chance of recovered node becoming susceptible: {rs}")
    
    initial = np.random.choice(range(len(g.node())))
    infected = [initial]
    susceptible = list(range(len(g.node())))
    susceptible.remove(initial)
    recovered = []
    sgraph = [len(susceptible)]
    igraph = [1]
    rgraph = [0]
    
    def color(n):
        if n in susceptible:
            return 'b'
        elif n in infected:
            return 'r'
        else:
            return 'g'
        
    for t in range(time):
        def si_calc(node):
            ni = [n for n in infected if n in g[node]]
            p = 1-((1-si)**len(ni))
            return p
        
        vul = [n for n in range(len(g.node())) if n in susceptible]
        if len(vul) > 0:
            si_calc = np.vectorize(si_calc)
            p_dist = si_calc(vul)
            
            factorsi = np.random.ranf(len(vul)) < p_dist
            infected_new = np.array(vul)[factorsi]
        else:
            infected_new = []
        
        factorir = np.where(np.random.ranf(len(infected)) < ir)
        recovered_new = list(np.array(infected)[factorir])
        
        factorrs = np.where(np.random.ranf(len(recovered)) < rs)
        susceptible_new = list(np.array(recovered)[factorrs])
     
        infected = list(filter(lambda n: n not in recovered_new, infected))
        recovered = list(filter(lambda n: n not in susceptible_new, recovered))
        susceptible = list(filter(lambda n: n not in infected_new, susceptible))
        
        infected.extend(infected_new)
        recovered.extend(recovered_new)
        susceptible.extend(susceptible_new)
        
        sgraph.append(len(susceptible))
        igraph.append(len(infected))
        rgraph.append(len(recovered))
        
        if len(infected) == 0:
            print(f'\nThe infection has been eradicated. Time taken: {t+1}')
            break
        
    colors = list(map(color, range(len(g.node()))))
    print(f"\nSIR model of graph after {t+1} time steps:")
    nx.draw(g, node_color = colors)
    plt.show()
    plt.clf()
    
    plt.plot(range(0, t+2), sgraph, color = 'b', label = 'Susceptible')
    plt.plot(range(0, t+2), igraph, color = 'r', label = 'Infected')
    plt.plot(range(0, t+2), rgraph, color = 'g', label = 'Recovered')
    plt.legend(loc = 'upper right')
    plt.xlabel('Time')
    plt.ylabel('Number of nodes')
    plt.show()

    return

g = nx.barabasi_albert_graph(1000, 20)

print(nx.info(g))
print(f"\nGlobal clustering coefficient: {nx.average_clustering(g)}")
print(f"Transitivity: {nx.transitivity(g)}")
nx.draw(g, node_color = 'y')
plt.show()
plt.clf()

degree_sequence = sorted([d for n, d in g.degree()])
count = Counter(degree_sequence)

plt.bar(count.keys(), count.values(), color='b')
plt.title("Degree Histogram")
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.show()
plt.clf()

sir(g, 0.1, 0.1, 0.1, 50)

    
    
    

    
