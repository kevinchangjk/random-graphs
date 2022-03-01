import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

g = nx.barabasi_albert_graph(100, 3)
pwd = nx.all_pairs_shortest_path_length(g)
pwds = []
for d in pwd:
    pwds.extend(d[1].values())

pwds = sorted(list(filter(lambda x: x != 0, pwds)))
count = Counter(pwds)
plt.bar(count.keys(), count.values())
plt.xlabel("Distance")
plt.ylabel("Number of pairs")
plt.title("Distance distribution")
plt.show()
