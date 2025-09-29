# =============================================
# Импорты необходимых библиотек
# =============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

# Настройка отображения графиков внутри ноутбука (если используется)
# plt.style.use('seaborn')  # опционально

# =============================================
# ЗАДАНИЕ 1: Открытие данных и визуализация распределения классов
# =============================================

# Шаг 1.1: Открытие данных
data_file_path = '../resources/spam_or_not_spam.csv'
data = pd.read_csv(data_file_path)

# Шаг 1.2: Расчёт количества спама и не спама
print("Первые 5 строк данных:")
print(data.head())
print("\nРазмер датасета:", data.shape)

# Подсчёт количества писем по меткам
label_counts = data['label'].value_counts().sort_index()
total = len(data)
spam_ratio = label_counts.get(1, 0) / total
ham_ratio = label_counts.get(0, 0) / total

print("\n=== ЗАДАНИЕ 1: Распределение классов ===")
print(f"Не спам (0): {label_counts.get(0, 0)} писем ({ham_ratio:.2%})")
print(f"Спам (1):    {label_counts.get(1, 0)} писем ({spam_ratio:.2%})")

# Шаг 1.3: Визуализация — столбчатая диаграмма
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='label', palette='Set2')
plt.title('Распределение спама и не спама')
plt.xlabel('Метка (0 = не спам, 1 = спам)')
plt.ylabel('Количество писем')
plt.xticks([0, 1], ['Не спам', 'Спам'])
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.show()

# =============================================
# ЗАДАНИЕ 2: Предобработка и векторизация
# =============================================

# Шаг 2.1: Замена пустых строк и строк из одних пробелов на NaN
# Регулярное выражение r'^\s*$' означает: начало строки, любые пробелы (включая ноль), конец строки
data['email'] = data['email'].replace(r'^\s*$', np.nan, regex=True)

# Удаление строк с пропущенными значениями в столбце 'email'
data_clean = data.dropna(subset=['email']).reset_index(drop=True)

print(f"\nПосле удаления пустых email: {data_clean.shape[0]} писем осталось из {data.shape[0]}")

# Шаг 2.2: Очистка текста (теперь безопасно, т.к. нет NaN)
data_clean['email'] = (
    data_clean['email']
    .str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)  # оставить только буквы, цифры, пробелы
    .str.replace(r'\s+', ' ', regex=True)             # нормализовать пробелы
    .str.strip()                                      # убрать пробелы по краям
    .str.lower()                                      # привести к нижнему регистру
)

# Шаг 2.3: Векторизация текста с помощью CountVectorizer (мешок слов)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_clean['email'])
y = data_clean['label']

# Шаг 2.4: Определение количества признаков (уникальных слов)
n_features = X.shape[1]
print(f"\n=== ЗАДАНИЕ 2: Векторизация завершена ===")
print(f"Размер матрицы признаков: {X.shape}")
print(f"Количество признаков (уникальных слов): {n_features}")

# =============================================
# ЗАДАНИЕ 3: Разделение данных и анализ целевой переменной
# =============================================

# Шаг 3.1: Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,        # сохраняем пропорции классов
    random_state=42    # для воспроизводимости
)

# Шаг 3.2: Среднее значение целевой переменной (доля спама в полном датасете)
mean_y = y.mean()
print(f"\n=== ЗАДАНИЕ 3: Информация о выборке ===")
print(f"Среднее значение целевой переменной (доля спама): {mean_y:.4f} ({mean_y:.2%})")
print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")

# =============================================
# ЗАДАНИЕ 4: Обучение модели с alpha=0.01 и оценка по метрикам
# =============================================

# Шаг 4.1: Обучение MultinomialNB с alpha=0.01
clf = MultinomialNB(alpha=0.01)
clf.fit(X_train, y_train)

# Шаг 4.2: Предсказания
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]  # вероятности для класса "спам" (1)

# Шаг 4.3: Расчёт метрик (не менее трёх + ROC-AUC)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n=== ЗАДАНИЕ 4: Оценка модели (alpha=0.01) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Дополнительно: подробный отчёт
print("\nПодробный отчёт по классам:")
print(classification_report(y_test, y_pred, target_names=['Не спам', 'Спам']))

# Шаг 4.4: Построение ROC-кривой
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================
# ЗАДАНИЕ 5: Подбор гиперпараметра alpha с помощью кросс-валидации
# =============================================

# Шаг 5.1: Определение сетки значений alpha
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# Шаг 5.2: Настройка GridSearchCV с 5-кратной кросс-валидацией
# Используем ROC-AUC как метрику (можно заменить на 'f1' или 'accuracy')
grid_search = GridSearchCV(
    estimator=MultinomialNB(),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',  # или 'f1' — зависит от задачи
    n_jobs=-1,
    verbose=1
)

# Шаг 5.3: Запуск поиска
print("\n=== ЗАДАНИЕ 5: Подбор оптимального alpha с помощью кросс-валидации ===")
print("Выполняется GridSearchCV...")

grid_search.fit(X_train, y_train)

# Шаг 5.4: Вывод результатов
print(f"\nЛучшее значение alpha: {grid_search.best_params_['alpha']}")
print(f"Лучший ROC-AUC на кросс-валидации: {grid_search.best_score_:.4f}")

# Шаг 5.5: Оценка лучшей модели на тестовой выборке
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

test_roc_auc_best = roc_auc_score(y_test, y_pred_proba_best)
print(f"ROC-AUC лучшей модели на тестовой выборке: {test_roc_auc_best:.4f}")

# Шаг 5.6: Визуализация зависимости качества от alpha
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(8, 5))
plt.plot(results['param_alpha'], results['mean_test_score'], marker='o')
plt.xscale('log')
plt.xlabel('alpha (логарифмическая шкала)')
plt.ylabel('Средний ROC-AUC (5-fold CV)')
plt.title('Зависимость качества от гиперпараметра alpha')
plt.grid(True)
plt.tight_layout()
plt.show()
