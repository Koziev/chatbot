# Модели для определения релевантности и выбора предпосылок

## Оценка качества по отдельной задаче и датасету

Сама по себе бинарная оценка релевантности двух предложений недостаточна
для использования модели в чатботе. Поскольку чатбот выбирает наиболее
релевантный факт для заданного вопроса из большого списка, то хочется,
чтобы модель присваивала парам предпосылка-вопрос пригодную к сравнению
оценку. Далее отранжировав пары по убыванию такой оценки, можно взять
топовую пару и далее использовать этот наиболее релевантный факт для
генерации ответа.

Чтобы проверить модели на такой задачи, в проекте есть специальный отдельный
датасет (см. файл [evaluate_relevancy.txt](https://github.com/Koziev/chatbot/blob/master/data/evaluate_relevancy.txt)).
В нем задаются наборы релевантных предпосылок и вопросов к ним. Например:

```
T: Лисичка живет в диком лесу
Q: Где обитает лиса
```

В ходе оценки к предпосылке "Лисичка живет в диком лесу" будут добавлены
все альтернативные предпосылки из файла premises. Затем модель вычисляет
релевантность каждой предпосылки в получающемся списке с каждым вопросом,
отмеченным в датасете префиксом Q:. Затем смотрим, является ли вычисленная
релевантность правильной пары максимальной среди всех прочих альтернатив.

Именно точность выбора релевантной предпосылки по данному датасету приводится
далее в столбце "eval".

## Baseline

В качестве baseline используется самый простой, но весьма эффективный для
данной задачи способ определения релевантности с помощью [коэффициента Жаккара](https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D1%8D%D1%84%D1%84%D0%B8%D1%86%D0%B8%D0%B5%D0%BD%D1%82_%D0%96%D0%B0%D0%BA%D0%BA%D0%B0%D1%80%D0%B0).
Оба предложения нарезаются на шинглы - перекрывающиеся n-граммы заданной длины (задается
в коде модели, принимается по умолчанию 3 символа).

Реализация на Python находится в файле [evaluate_other_metrics.py](https://github.com/Koziev/chatbot/blob/master/PyModels/evaluate_other_metrics.py). Запуск
расчета с указанием необходимых параметров выполнен в скрипте [eval_jaccard_relevancy.sh](https://github.com/Koziev/chatbot/blob/master/scripts/eval_jaccard_relevancy.sh).

## Градиентный бустинг и шинглы

В проекте есть две альтернативные реализации детектора релевантности на
базе градиентного бустинга.

[lgb_relevancy.py](https://github.com/Koziev/chatbot/blob/master/PyModels/lgb_relevancy.py) - тренировка бинарного классификатора LightGBM
[xgb_relevancy.py](https://github.com/Koziev/chatbot/blob/master/PyModels/xgb_relevancy.py) - тренировка бинарного классификатора XGBoost

В обоих случаях оба входных предложения представляются двумя множествами
шинглов заданной длины (параметр командной сроки --shingle_len, по умолчанию 3).

В скриптах [train_xgb_relevancy.cmd](https://github.com/Koziev/chatbot/blob/master/scripts/train_xgb_relevancy.cmd),
[train_xgb_relevancy.sh](https://github.com/Koziev/chatbot/blob/master/scripts/train_xgb_relevancy.sh), [train_lgb_relevancy.sh](https://github.com/Koziev/chatbot/blob/master/scripts/train_lgb_relevancy.sh)
можно найти примеры запуска обучения моделей. Результат обучения - файлы с именами 
lgb_relevancy.model (бустер, сохраненный в LightGBM) и lgb_relevancy.config (данные для
подготовки признаков текста), аналогично для XGBoost модели.

Расчет оценки качества выбора релевантной предпосылки выполняется программами
тренировки при задании опции --run_mode evaluate. Примеры запуска расчета
можно найти в скриптах [eval_xgb_relevancy.sh](https://github.com/Koziev/chatbot/blob/master/scripts/eval_xgb_relevancy.sh)
и [eval_lgb_relevancy.sh](https://github.com/Koziev/chatbot/blob/master/scripts/eval_lgb_relevancy.sh).

## Консольный тест моделей релевантности

С помощью консольной программы [find_premise_for_question.py](https://github.com/Koziev/chatbot/blob/master/PyModels/find_premise_for_question.py)
можно вручную проверить работу натренированных моделей:




## Прочие необучаемые и обучаемые без учителя метрики

...to be written...


## Нейросетевые модели

...to be written...


