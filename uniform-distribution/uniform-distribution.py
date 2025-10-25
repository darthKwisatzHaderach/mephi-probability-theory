import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import bernoulli


# Моделируем 200 запусков
data = bernoulli.rvs(size=200, p=0.02)
# Визуализация
ax = sns.displot(data, kde=False, color='seagreen')
ax.set(xlabel="Значение случайной величины (0 - проиграл, 1 - выиграл)", ylabel="Частота")
plt.show()


p = sum(st.binom.pmf(k, 2, 0.15) for k in range(0, 2))
print(p)  # ≈ 0.91