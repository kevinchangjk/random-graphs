import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

def sir(g, si, ir, ini, time):
    print(f"\nInfection has begun, with initial probability of infection being {ini}.")
    print(f"Chance of susceptible node becoming infected: {si}")
    print(f"Chance of infected node recovering: {ir}")
    
    size = len(g.nodes())
    
    inin = np.random.ranf(size) < ini
    state = inin * 1
    i_plot = [sum(state)]
    r_plot = [0]
    s_plot = [size-i_plot[0]]
    
    def color(n):
        if state[n] == 0:
            return 'b'
        elif state[n] == 1:
            return 'r'
        else:
            return 'g'
    
    def infect():
        base = nx.adjacency_matrix(g)
        infected = state == 1
        infected_p = base * infected
        infected_pdist = np.random.ranf(size) < 1-(1-si)**infected_p
        infected_new1 = np.where(infected_pdist ==1)[0]
        vulnerable = np.where(state == 0)[0]
        infected_new = np.array(list(filter(lambda x: x in vulnerable, infected_new1)))
        
        state_new = state.copy()
        if np.size(infected_new) > 0:
            state_new[infected_new] = 1
        recovery = np.random.ranf(size) < ir
        recovered = [i if recovery[i] == 1 and state[i] == 1 else -1 for i in range(size)]
        recovered = np.array(list(filter(lambda x: x >= 0, recovered)))
        if np.size(recovered) > 0:
            state_new[recovered] = -1
        
        return state_new
    
    for t in range(1, time+1):
        state = infect()
        state_count = Counter(state)
        s_plot.append(state_count[0])
        i_plot.append(state_count[1])
        r_plot.append(state_count[-1])
        if state_count[1] == 0:
            print(f'\nThe infection has been eradicated. Time taken: {t}')
            break
        
    colors = list(map(color, range(size)))
    print(f"\nSIR model of graph after {t} time steps:")
    nx.draw(g, node_color = colors)
    plt.show()
    plt.clf()
    
    plt.plot(range(0, t+1), s_plot, color = 'b', label = 'Susceptible')
    plt.plot(range(0, t+1), i_plot, color = 'r', label = 'Infected')
    plt.plot(range(0, t+1), r_plot, color = 'g', label = 'Recovered')
    plt.legend(loc = 'upper right')
    plt.xlabel('Time')
    plt.ylabel('Number of nodes')
    plt.show()
        
    return 
        
g = nx.barabasi_albert_graph(100, 6)

sir(g, 0.3, 0.4, 0.04, 50)