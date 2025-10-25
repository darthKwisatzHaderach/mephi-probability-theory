import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(7, 7))

# Рисуем квадрат
plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=1.5)

# Заливаем область встречи (светло-голубая)
x_fill = np.linspace(0, 1, 500)
y_fill = np.linspace(0, 1, 500)
X, Y = np.meshgrid(x_fill, y_fill)
meet = np.abs(X - Y) <= 0.25
plt.contourf(X, Y, meet, levels=[0.5, 1], colors=['#cce5ff'], alpha=0.6)

# --- Верхняя граница: y = x + 1/4 ---
x_upper = np.linspace(0, 0.75, 100)
y_upper = x_upper + 0.25
plt.plot(x_upper, y_upper, 'r--', linewidth=1.8, label=r'$y = x + \frac{1}{4}$')

# --- Нижняя граница: y = x - 1/4 ---
x_lower = np.linspace(0.25, 1, 100)
y_lower = x_lower - 0.25
plt.plot(x_lower, y_lower, 'r--', linewidth=1.8, label=r'$y = x - \frac{1}{4}$')

# Точки пересечения
points = [(0, 0.25), (0.75, 1), (0.25, 0), (1, 0.75)]
for (px, py) in points:
    plt.plot(px, py, 'bo', markersize=6)
    plt.text(px + 0.02, py + 0.02, f'({px:g}, {py:g})', fontsize=11, color='darkblue')

# === Подписи площадей (чёрным цветом) ===

# Верхний треугольник
upper_center_x = (0 + 0 + 0.75) / 3
upper_center_y = (0.25 + 1 + 1) / 3
plt.text(
    upper_center_x, upper_center_y,
    r'$S_1 = \frac{1}{2} \cdot \frac{3}{4} \cdot \frac{3}{4} = \frac{9}{32}$',
    fontsize=12, color='black', ha='center', va='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
)

# Нижний треугольник
lower_center_x = (0.25 + 1 + 1) / 3
lower_center_y = (0 + 0 + 0.75) / 3
plt.text(
    lower_center_x, lower_center_y,
    r'$S_2 = \frac{1}{2} \cdot \frac{3}{4} \cdot \frac{3}{4} = \frac{9}{32}$',
    fontsize=12, color='black', ha='center', va='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
)

# === Настройки графика ===
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.xlabel('$x$ (время прихода первого)', fontsize=12)
plt.ylabel('$y$ (время прихода второго)', fontsize=12)
plt.title('Задача 1.22', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='upper left')

plt.show()