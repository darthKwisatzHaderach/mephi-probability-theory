import numpy as np
import pandas as pd

data_file_path = '../resources/spam_or_not_spam.csv'
data = pd.read_csv(data_file_path)
print(data.head())

count = data.groupby('label').count()
print(count)
print(data.shape)

data_clean = data.copy()

# Что такое .str?
# .str - это аксессор (accessor) в pandas, который предоставляет строковые методы для работы с элементами Series/DataFrame.
# Когда у вас есть столбец в DataFrame (Series), который содержит строки, .str позволяет применять строковые методы к каждому элементу столбца.
# df['text'].strip()      ❌ Ошибка! Нельзя применить к целому столбцу
# df['text'].str.strip()  ✅ Применяет strip() к каждой строке в столбце

# Заменить NaN на пустые строки, чтобы в дальнейшем не возникло проблем в шаге vocabulary = list(set(train_data['email'].sum())),
# где происходит работа со списками. Метод .str.split() не превращает NaN в список, а оставляет как есть (float-тип nan).
data_clean['email'] = data_clean['email'].fillna('')

# Очистка текста в столбце 'email' от всех не-алфавитно-цифровых символов
data_clean['email'] = data_clean['email'].str.replace(r'\W+', ' ', regex=True)

# Нормализация пробелов в email-адресах в столбце 'email'
data_clean['email'] = data_clean['email'].str.replace(r'\s+', ' ', regex=True).str.strip()

data_clean['email'] = data_clean['email'].str.lower()
data_clean['email'] = data_clean['email'].str.split()

print(data_clean.head())

print(data_clean['label'].value_counts())
print(data_clean.shape)

# Смотрим на процентное соотношение двух классов писем СПАМ (spam) и НЕ СПАМ (ham)
print(data_clean['label'].value_counts() / data_clean.shape[0] * 100)

# Данные несбалансированны - большинство сообщений не являются спамом.
# Это важная информация, которую нужно учитывать при построении и оценке модели классификации.

# Разделение на обучающую и тестовую выборки, сохранив пропорции классов.
# Выберем случайную выборку из DataFrame
# frac=0.8 - берет 80% от всех данных
# random_state=42 - обеспечивает воспроизводимость результата
train_data = data_clean.sample(frac=0.8, random_state=42)

# Создаем тестовую выборку путем удаления строк, которые попали в обучающую выборку.
# train_data.index - индексы строк, которые уже в обучающей выборке
# drop() - удаляет эти строки из исходных данных
test_data = data_clean.drop(train_data.index)

# Сбрасываем индекс обучающей выборки
# drop=True - удаляет старый индекс (не создает новый столбец)
# Создает новый последовательный индекс (0, 1, 2, ...)
train_data = train_data.reset_index(drop=True)

# Сбрасываем индекс для тестовой выборки
test_data = test_data.reset_index(drop=True)

# Теперь проверяем аналогичное соотношение классов в обучающей выборке
print(train_data['label'].value_counts() / train_data.shape[0] * 100)
print(train_data.shape)

# Аналогично для тестовой выборки
print(test_data['label'].value_counts() / test_data.shape[0] * 100)
print(test_data.shape)

# На данном этапе получено разное распределение меток в train и test выборках:
# Train: 83.96% ham, 16.04% spam
# Test: 80.83% ham, 19.17% spam
# Разница в распределении спама: 16.04% vs 19.17% (разница ≈ 3%)
# Почему это проблема:
# 1. Смещение выборки (Sampling Bias)
# Модель обучается на одном распределении, а тестируется на другом. Это может исказить результаты.
# 2. Нерепрезентативная оценка
# Если в test выборке больше спама, то:
#  - Метрики для класса "спам" могут быть завышены/занижены
#  - Реальная производительность модели будет другой
# 3. Проблема с редким классом
# Спам - меньшинство (16-19%). Небольшие изменения в распределении сильно влияют на оценку качества.
# Насколько это критично:
# Умеренно критично. Разница в 3% не катастрофична, но требует внимания.
# Пороги критичности:
#  < 2% разницы - обычно приемлемо
#  2-5% разницы - требует проверки стабильности модели
#  > 5% разницы - серьезная проблема

# Рекомендую переразделить данные со стратификацией.
# Это стандартная практика для задач классификации, особенно с несбалансированными данными.
# Стратификация (Stratification) - это техника разделения данных, которая сохраняет исходное распределение классов в train и test выборках.

# Есть библиотеки для выполнения стратификации
# Пример:
# from sklearn.model_selection import train_test_split
#
# Стратификация по метке 'label'
# train_data, test_data = train_test_split(
#     data_clean,
#     test_size=0.2,
#     random_state=42,
#     stratify=data_clean['label']  # ← Вот это важно!
# )

# Ручная балансировка если нельзя использовать train_test_split
ham_data = data_clean[data_clean['label'] == 0]
spam_data = data_clean[data_clean['label'] == 1]

# Разделяем каждый класс отдельно
ham_train = ham_data.sample(frac=0.8, random_state=42)
spam_train = spam_data.sample(frac=0.8, random_state=42)

train_data = pd.concat([ham_train, spam_train])
test_data = data_clean.drop(train_data.index)

# Теперь проверяем соотношение классов в обучающей выборке
print(train_data['label'].value_counts() / train_data.shape[0] * 100)
print(train_data.shape)

# Аналогично для тестовой выборки
print(test_data['label'].value_counts() / test_data.shape[0] * 100)
print(test_data.shape)

# Исходные данные остаются нетронутыми
train_emails = train_data['email']      # список списков слов
train_labels = train_data['label']      # Series 0/1

# Словарь — со всеми словами, включая 'email', 'label'
vocabulary = list(set(word for email in train_emails for word in email))
print(f"Размер словаря: {len(vocabulary)}")
print("Примеры слов:", vocabulary[11:20])

# Создаём матрицу признаков ОТДЕЛЬНО
X_train = pd.DataFrame(
    [[email_words.count(word) for word in vocabulary] for email_words in train_emails],
    columns=vocabulary
)

# Для каждого email-сообщения посчитаем, сколько раз в нём встречается каждое слово.
word_counts_per_email = pd.DataFrame([
    [row[0].count(word) for word in vocabulary]
    for _, row in train_data.iterrows()], columns=vocabulary)
print(word_counts_per_email.head())

# Добавим частоты каждого слова в обучающий датасет.
train_data = pd.concat([train_data, word_counts_per_email], axis=1)
print(train_data.head())

# Посчитаем необходимые значения для формулы Байеса. Нормировочный коэффициент возьмем равным 1
alpha = 1
Nvoc = len(vocabulary)
Pspam = train_labels.value_counts()[1] / train_data.shape[0]      # отношение всех писем со спамом к общему числу сообщений
Pham = train_labels.value_counts()[0] / train_data.shape[0]        # аналогично для не спама
Nspam = train_data.loc[train_labels == 1, 'email'].apply(len).sum() # число уникальных слов в спаме
Nham = train_data.loc[train_labels == 0, 'email'].apply(len).sum()   # число уникальных слов в НЕ спаме

# .loc — это специальный аксессор в pandas для выборки данных по меткам (label-based indexing).
# df.loc[строки, столбцы]
# строки — условие или список индексов строк
# столбцы — имя столбца или список имён
# Пример:
# train_data.loc[train_data['label'] == 1, 'hello']
# Берёт все значения в столбце 'hello', но только для тех строк, где label == 1 (т.е. только для спама).

# Рассчитываем вероятности того, что если слово встречается - это спам или не спам
def p_w_spam(word):
    if word in train_data.columns:
        return (X_train.loc[train_labels == 1, word].sum() + alpha) / (Nspam + alpha * len(vocabulary))
    else:
        return 1

def p_w_ham(word):
    if word in train_data.columns:
        return (X_train.loc[train_labels == 0, word].sum() + alpha) / (Nspam + alpha * len(vocabulary))
    else:
        return 1

# Определяем вероятности спам/не спам.
# Проверяем, если вероятность того, что это не спам > это спам => не спам и наоборот.
# Если вероятности равны, то выдаем в лог информацию о некорректной классификации
def classify(message):
    p_spam_given_message = Pspam
    p_ham_given_message = Pham
    for word in message:
        p_spam_given_message *= p_w_spam(word)
        p_ham_given_message *= p_w_ham(word)
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'классификация некорректна'

# Используем тестовые данные
test_data['predicted'] = test_data['email'].map(classify)
print(test_data.head())

# Оценим долю сообщений, которые определены правильно
correct = (test_data['predicted'] == test_data['label']).sum() / test_data.shape[0]
print(f"Правильных предсказаний {correct * 100:3f} %")
