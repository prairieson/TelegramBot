import random


# Словарь
intents = {
    'hello': {
        'examples': ['hello', "Привет", "Здравствуйте"],
        'responses': ['Добрый день!', "Как дела?", "Как настроение?"]
    },
    'weather': {
        'examples': ['Какая погода?', 'Что за окном', "Во что одеваться?"],
        'responses': ['Погода отличная!', "У природы нет плохой погоды!"],
    }
}
# input = ввод данных от пользователя
# random.choice = выбор случайного элемента из списка
# print = вывод на экран


text = input()
for intent_name in intents:
    for example in intents[intent_name]['examples']:
        if text == example:
            answer = random.choice(intents[intent_name]['responses'])
            print(answer)
