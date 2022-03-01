import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

def edge_get(num):
    with open(f"{num}.edges") as f:
        edges = f.read().split("\n")
        edges.pop()
        edges = [tuple(map(int, edge.split(" "))) for edge in edges]
        edges_add = list(set([(num, i[j]) for i in edges for j in range(1)]))
        edges.extend(edges_add)
    return edges

with open("facebook_combined.txt") as f:
    edges = f.read().split("\n")
    edges.pop()
    

    