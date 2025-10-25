import networkx as nx
import matplotlib.pyplot as plt

# Создаём K_{3,3}
G = nx.complete_bipartite_graph(3, 3)

# Переименуем вершины в A, B, C, D, E, F
mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
G = nx.relabel_nodes(G, mapping)

# Рисуем с двумя колонками (бипартитный layout)
pos = nx.bipartite_layout(G, nodes=['A', 'B', 'C'])

plt.figure(figsize=(6, 5))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_weight='bold')
plt.title(r"3-регулярный граф на 6 вершинах: $K_{3,3}$")
plt.show()