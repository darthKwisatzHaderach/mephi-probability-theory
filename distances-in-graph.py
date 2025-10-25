import networkx as nx
import pandas as pd
# строим граф
G = nx.Graph()
G.add_edges_from([
    ('A','B'),
    ('B','C'), ('B','D'),
    ('C','E'), ('C','F'),
    ('D','E'),
    ('E','F'),
])
# матрица смежности
A = nx.to_pandas_adjacency(G, nodelist=['A','B','C','D','E','F'])
print(A)
# эксцентриситеты, радиус и диаметр
ecc = nx.eccentricity(G)
radius = nx.radius(G)
diameter = nx.diameter(G)
print(ecc, radius, diameter)