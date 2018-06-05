# Вопросно-ответная диалоговая система

## Обучение

1. Необходимо подготовить word2vector модели, например с помощью [w2v.py](https://github.com/Koziev/Word2Vec/blob/master/PyUtils/w2v.py).
Модели в составе чат-бота используют разные варианты встраиваний, согласно
опциям в файлах scripts/train_*.sh. На данный момент нужны следующие модели:

w2v.CBOW=1_WIN=5_DIM=8.bin   
w2v.CBOW=1_WIN=5_DIM=32.bin   

Параметры моделей указаны в именах файлов для удобства.

2. Обучение моделей чатбота:

train_wordchar2vector.sh   
train_nn_wordcopy3.sh   
train_nn_model_selector.sh   
train_xgb_person_classifier.sh   
train_xgb_yes_no.sh   
train_lgb_relevancy.sh   


## Консольный фронтенд для бота

Реализован в файле [test_simple_console_answering_machine.py](https://github.com/Koziev/chatbot/blob/master/PyModels/bot/test_simple_console_answering_machine.py).
Запуск выполняется скриптом scripts/console_bot.sh


![Console frontend for chatbot](chatbot-console.PNG)

## Чатбот для Telegram

Реализован в файле [test_telegram_bot.py](https://github.com/Koziev/chatbot/blob/master/PyModels/bot/test_telegram_bot.py)

![Telegram frontend for chatbot](chatbot-telegram.png)


## Тренировка и использование модели посимвольного встраивания слов

Описание архитектуры модели и ее тренировки смотрите на [отдельной странице](./PyModels/trainers/README.wordchar2vector.md).

## Модель для определения релевантности факта и вопроса

См. [описание на отдельной странице](README.relevance.md).
