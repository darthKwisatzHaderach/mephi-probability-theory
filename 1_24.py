import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('dataset_diseases.csv')

# Проверка структуры данных
print(df.head())
print(df['Status'].value_counts())

# Кодирование категориальных признаков в числовые метки
# Наивный Байес (CategoricalNB) требует целочисленные входные данные ≥ 0

# Создаём копию для признаков
X = pd.DataFrame()

# Кодируем 'Test' (Positive/Negative → 0/1 или наоборот)
le_test = LabelEncoder()
X['Test'] = le_test.fit_transform(df['Test'])

# Кодируем 'Age_Group' (Young/Old → 0/1)
le_age = LabelEncoder()
X['Age_Group'] = le_age.fit_transform(df['Age_Group'])

# Кодируем целевую переменную 'Status' (Infected/Not_infected → 0/1)
le_status = LabelEncoder()
y = le_status.fit_transform(df['Status'])

# Разделение данных на обучающую и тестовую выборки (80% / 20%)
# stratify=y — сохраняет пропорции классов в обеих выборках
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Создание и обучение модели Наивного Байеса для категориальных данных
model = CategoricalNB()
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка качества модели

# Точность (accuracy) — доля правильных предсказаний
accuracy = accuracy_score(y_test, y_pred)

# Подробный отчёт по метрикам (precision, recall, f1-score)
report = classification_report(y_test, y_pred, target_names=le_status.classes_)

# Матрица ошибок (confusion matrix)
cm = confusion_matrix(y_test, y_pred)

# Вывод результатов
print("=== Результаты работы модели Наивного Байеса ===\n")
print(f"Точность (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
print("Подробный отчёт по классам:")
print(report)
print("Матрица ошибок:")
print(cm)
