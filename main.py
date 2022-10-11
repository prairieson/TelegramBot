"""
 {Вопрос на входе} => {Алгоритм ответа} => {Ответ на выходе}
 Простой алгоритм – простой поиск по базе известных вопросов и ответов
"""
import urllib.request  # Для работы со ссылками
import json  # Для работы с json
import random
from sklearn.feature_extraction.text import CountVectorizer  # Для обучения векторайзера (простенького ML)
from sklearn.neural_network import MLPClassifier

"""
ШАГ 1. Собираем данные, на основании которых будем обучать.

1. Фраза на вход => Модель предсказывает интент фразы.
2. Входные данные (Фразы, X) Выходные данные (Интенты, y).
3. Модель обучится на наших примерах и сможет предсказывать интенты по фразе.
"""
# Скачаем файл с большим количеством примеров для обучения
url = "https://drive.google.com/uc?export=download&id=1VyvE5DGPqWFBHm5MFPUv-upq-SMZEXHx"
filename = "local_intents_dataset.json"
urllib.request.urlretrieve(url, filename)

# Считываем файл в словарь
with open(filename, 'r', encoding='UTF-8') as file:
    data = json.load(file)

"""
ШАГ 2. Преобразуем слова в числа, набор фраз в вектор.

1. На вход подается большой набор фраз.
2. Векторайзер обучается (каждому слову выделяется отдельная ячейка, 
    куда будет вставляться код - число, сколько раз слово встретилось во фразе).
3. Векторайзер готов работать с новыми текстами.
"""
X = []
y = []
for name in data:
    for phrase in data[name]['examples']:
        X.append(phrase)
        y.append(name)
    for phrase in data[name]['responses']:
        X.append(phrase)
        y.append(name)

local_vectorize = CountVectorizer()
local_vectorize.fit(X)
X_vec = local_vectorize.transform(X)

"""
ШАГ 3. Обучаем классификатор текстов.
"""
# Создаём модель
model = MLPClassifier()
# Обучаем модель
model.fit(X_vec, y)

"""
ШАГ 4. Оцениваем качество.

Accuracy = (n угаданных/N всего) — доля правильных ответов (больше - лучше).
"""
model.score(X_vec, y)  # Качество на тренировочной выборке (accuracy)


"""
ШАГ 5. Пишем функцию для получения интента с помощью ML.
"""


def get_intent(text):
    text_vec = local_vectorize.transform([text])
    intent = model.predict(text_vec)[0]
    return intent


def get_response(intent):
    return random.choice(data[intent]['responses'])


def bot(phrase):
    intent = get_intent(phrase)
    answer = get_response(intent)
    return answer


# Запуск, чтобы пообщаться
text = ""
while get_intent(text) != 'bye':
    text = input('< ')
    print('> ', bot(text))
