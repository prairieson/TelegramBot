"""
 {Вопрос на входе} => {Алгоритм ответа} => {Ответ на выходе}
 Простой алгоритм – простой поиск по базе известных вопросов и ответов
"""
import urllib.request  # Для работы со ссылками
import json  # Для работы с json
import random
from sklearn.feature_extraction.text import CountVectorizer  # Для обучения векторайзера (простенького ML)
import nest_asyncio
from telegram import Update  # update for get information from server (new messages, new contacts)
from telegram.ext import ApplicationBuilder  # Tool for creating and customizing an application
from telegram.ext import MessageHandler  # Tool to respond to user action
from telegram.ext import filters
import pickle

"""
ШАГ 1. Загружаем обученную модель.
"""
# Скачаем файл с обученным сетом
url = "https://drive.google.com/uc?export=download&id=1zXL6iGZKgJGQPv_IO1Dqgt_vIkyewBJD"
filename = "model.sav"
urllib.request.urlretrieve(url, filename)

# С помощью pickle распаковываю модель
with open(filename, 'br') as file:
    model = pickle.load(file)

"""
ШАГ 2. Собираем данные, на основании которых будем отвечать.
"""
# Скачаем файл с большим количеством примеров для обучения
url = "https://drive.google.com/uc?export=download&id=1VyvE5DGPqWFBHm5MFPUv-upq-SMZEXHx"
filename = "local_intents_dataset.json"
urllib.request.urlretrieve(url, filename)

# Считываем файл в словарь
with open(filename, 'r', encoding='UTF-8') as file:
    data = json.load(file)

"""
ШАГ 3. Преобразуем слова в числа, набор фраз в вектор.
Это нужно для того, чтобы обученная модель могла выбирать из чего ей отвечать.
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
ШАГ 4. Оцениваем качество.
Это нужно при запуске для контроля качества.
Accuracy = (n угаданных/N всего) — доля правильных ответов (больше - лучше).
"""
model.score(X_vec, y)  # Качество на тренировочной выборке (accuracy)

"""
ШАГ 5. Пишем функцию для получения интента с помощью ML.
"""


def get_intent(element):
    text_vec = local_vectorize.transform([element])
    intent = model.predict(text_vec)[0]
    return intent


def get_response(intent):
    return random.choice(data[intent]['responses'])


def bot(phrase_from):
    intent = get_intent(phrase_from)
    answer = get_response(intent)
    return answer

"""
ШАГ 6. Put the secret_file.txt file with the token from the bot in the project folder.
"""
with open('secret_file.txt', 'r') as f:
    TOKEN = f.read()


"""
ШАГ 7. Обучаем бота взаимодействовать.

1. Фраза на вход => Модель предсказывает интент фразы.
2. Входные данные (Фразы, X) Выходные данные (Интенты, y).
3. Модель обучится на наших примерах и сможет предсказывать интенты по фразе.
"""
nest_asyncio.apply()  # Для того чтобы функция bot могла работать асинхронно: для нескольких пользователей сразу


# Функция для MessageHandler'a: вызывать её при каждом сообщении бота
async def reply(update: Update, context) -> None:
    user_text = update.message.text
    reply_for_user = bot(user_text)
    print('<', user_text)
    print('>', reply_for_user)

    pass  # Ответ пользователю

    await update.message.reply_text(reply_for_user)


# Создаём объект приложения – связываем его с токеном
app = ApplicationBuilder().token(TOKEN).build()

# Создаём обработчик текстовых сообщений
handler = MessageHandler(filters.Text(), reply)

# Добавляем обработчик в приложение
app.add_handler(handler)

# Запускаем приложение: бот крутится
app.run_polling()
