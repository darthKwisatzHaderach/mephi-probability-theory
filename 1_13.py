# Задача 1.13 (1 балл)
# Дана булева формула x_1 →(x_2 →(x_3 →(⋯ → x_10))) (всего 9 вложенных импликаций).
# Сколько существует наборов значений переменных, на которых формула принимает значение 0.

import pandas as pd
import itertools

print("Наборы, где формула ложна (результат = 0):")
print("x1 x2 x3 x4 x5 x6 x7 x8 x9 x10")

# Генерируем все 1024 комбинации для x1 ... x10
rows = []
for bits in itertools.product([0, 1], repeat=10):
    # bits = (x1, x2, ..., x10)
    # Формула: (not x1) or (not x2) or ... or (not x9) or x10
    result = any(not b for b in bits[:9]) or bits[9]
    row = {f"x{i+1}": bits[i] for i in range(10)}
    row["result"] = int(result)
    rows.append(row)
    if not result:  # то есть result == False (или 0)
        print(" ".join(str(b) for b in bits))

# Создаём DataFrame
df = pd.DataFrame(rows)

# Опционально: переупорядочить столбцы
cols = [f"x{i}" for i in range(1, 11)] + ["result"]
df = df[cols]

# Выводим первые и последние несколько строк
print("Первые 5 строк:")
print(df.head())
print("\nПоследние 5 строк:")
print(df.tail())

# Опционально: сохранить в CSV
# df.to_csv("implication_10_vars.csv", index=False)