# Модели для определения релевантности и выбора предпосылок

Определение релевантности двух предложений является основной целого пласта методов
и подзадач в NLP, информацию о которых можно найти по ключевым терминам "Natural Language Sentence Matching",
"machine comprehension", "answer sentence selection", "natural language inference" и т.д.
Например, см. описание модели в [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/abs/1702.03814).

## Оценка качества по отдельной задаче и датасету

Сама по себе бинарная оценка релевантности двух предложений недостаточна
для использования модели в чатботе. Поскольку чатбот выбирает наиболее
релевантный факт для заданного вопроса из большого списка, то хочется,
чтобы модель присваивала парам предпосылка-вопрос пригодную к сравнению
оценку (rank). Далее отранжировав пары по убыванию такой оценки, можно взять
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

На данный момент максимальную точность выбора факта обеспечивается моделью на базе LightGBM.

## Консольный тест моделей релевантности

С помощью консольной программы [find_premise_for_question.py](https://github.com/Koziev/chatbot/blob/master/PyModels/find_premise_for_question.py)
можно вручную проверить работу натренированных моделей:

```
:> кто ловит мышей


0.9907   кошка ловит мышку
0.9604   лисы ловят мышей
0.9463   змеи ловят мышей
0.3596   коровы не ловят мышей
0.2656   мышка любит сыр
0.0384   лисы охотятся на мышей
0.0328   у мышей нет крыльев
0.0278   мышь не хищник
0.0278   на мышей охотятся лисы и кошки
0.0157   коровы мычат
```

Список фактов программа читает из файлов ../data/premises.txt и ../data/premises_1s.txt.

До некоторой степени одна модель релевантности может справляться даже
с вопросами, которые требуют смены грамматического лица для правильного
сопоставления с фактами в базе знаний, что обеспечивается другими
тренируемыми моделями:

:> как тебя зовут

```
0.7584   меня зовут кеша
0.1165   мне один год
0.0533   я тебя не вижу
0.0492   я тебя слышу
0.0357   мой возраст один год
0.0196   я тебе верю
0.0125   я понимаю тебя
0.0116   я не вру
0.0081   я робот
```


## Прочие необучаемые и обучаемые без учителя метрики

...to be written...


## Нейросетевые модели

### Реализация Skip-Thoughts на Keras

Исходный текст: https://github.com/Koziev/chatbot/blob/master/PyModels/train_skip_thoughts.py

За основу взята идея из [Skip-Thoughts](https://github.com/ryankiros/skip-thoughts). Основное
достоинство модели - unsupervised режим тренировки, поэтому нет необходимости в трудоемкой
разметке обучающих данных.

Отличия от оригинала:

1) Реализовано на Keras.

2) Тренируемся предсказывать только последующее предложение, а не предыдущее и последующее.

3) Слова представлены фиксированными word2vec эмбеддингами, они не меняются в ходе обучения,
так как это ухудшает точность.

4) Есть early stopping по валидационному набору, поэтому обучение идет не заданное
заранее число эпох.

5) Размерность векторов слов и предложений значитально меньше, чем в оригинальной модели.

Модель представляет из себя простую рекуррентную sequence 2 sequence сетку. Предложения
берутся из большого русскоязычного корпуса с художественной литературой и всякой публицистикой.

Результат работы программы - файлы train_skip_thoughts.arch, train_skip_thoughts.weights
и train_skip_thoughts.config, в которых сохранены параметры и веса кодирующей
части сетки. Сохраненная модель позволяет для цепочки слов, представленных
векторами word2vec, получить вектор, чтобы далее натренировать, к примеру,
модель второго уровня для классификации пар предложения на релевантные и нерелевантные - см.
исходный текст в [nn_relevancy_skip_thoughts.py](https://github.com/Koziev/chatbot/blob/master/PyModels/nn_relevancy_skip_thoughts.py).

### Нейросетевой классификатор релевантности

...to be written...


# Полезные ссылки

1. [aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016_yang.pdf)  
2. [Deep Learning for Natural Language Processing: Theory and Practice](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/CIKM14_tutorial_HeGaoDeng.pdf)  
3. [A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval ](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf)  
4. [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/abs/1702.03814)  
