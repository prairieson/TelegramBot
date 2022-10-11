"""
 {Вопрос на входе} => {Алгоритм ответа} => {Ответ на выходе}
 Простой алгоритм – простой поиск по базе известных вопросов и ответов
 """
 import urllib.request  # Для работы со ссылками
 import json  # Для работы с json
 from sklearn.feature_extraction.text import CountVectorizer  # Для обучения векторайзера (простенького ML)


 # Скачаем файл с большим количеством примеров для обучения
 url = "https://drive.google.com/uc?export=download&id=1VyvE5DGPqWFBHm5MFPUv-upq-SMZEXHx"
 filename = "local_intents_dataset.json"
 urllib.request.urlretrieve(url, filename)


 # Считываем файл в словарь
 with open(filename, 'r', encoding='UTF-8') as file:
     data = json.load(file)


 """
 Пробегаем по всему словарю – берём пары ключ-значение и раскладываем по двум спискам.
 Сначала берём пары name, intent.
 Потом для каждого элемента в examples и responses кладём в массив X элемент,
 Тем временем в y кладём name – название корневого интента.
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