import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Матрица смежности
adj_matrix = np.array([
    [0, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0]
])

# Создаём граф из матрицы
G = nx.from_numpy_array(adj_matrix)

# Переименуем вершины: 0->A, 1->B, ..., 5->F
mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
G = nx.relabel_nodes(G, mapping)

# Определяем компоненты связности
components = list(nx.connected_components(G))

# Назначаем цвета вершинам по компонентам
color_map = []
for node in G.nodes():
    for i, comp in enumerate(components):
        if node in comp:
            color_map.append(f'C{i}')  # используем стандартную палитру matplotlib
            break

# Визуализация
plt.figure(figsize=(8, 6))
pos = nx.circular_layout(G)
nx.draw(
    G, pos,
    with_labels=True,
    node_color=color_map,
    node_size=800,
    font_size=16,
    font_weight='bold',
    edge_color='gray',
    linewidths=1.5,
    alpha=0.9
)
plt.title("Граф по матрице смежности", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
