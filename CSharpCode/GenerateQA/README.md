# Утилита для генерации троек ПРЕДПОСЫЛКА-ВОПРОС-ОТВЕТ

Вспомогательная утилита для чат-бота https://github.com/Koziev/chatbot, реализованная
на C# как консольная утилита для MS Windows.

Генерирует датасет с тройками предложение ПРЕДПОСЫЛКА-ВОПРОС-ОТВЕТ, например:

```
T: Памперсы имеют индикатор влагонасыщения
Q: памперсы имеют что?
A: индикатор влагонасыщения

T: Памперсы имеют индикатор влагонасыщения
Q: памперсы что имеют?
A: индикатор влагонасыщения
```

Больше примеров можно увидеть тут https://github.com/Koziev/NLP_Datasets/blob/master/QA/premise_question_answer4.txt
 
Используются результаты работы утилиты https://github.com/Koziev/chatbot/tree/master/CSharpCode/ExtractFactsFromParsing
 
Для морфологического разбора фактов используется API Грамматического Словаря (http://solarix.ru/api/ru/list.shtml),
исходные тексты на C# и C++ лежат в отдельном репозитории здесь https://github.com/Koziev/GrammarEngine/tree/master/src/demo/ai/solarix/engines
 
Частеречная разметка требует наличия русской словарной базы. Можно взять готовые файлы здесь https://github.com/Koziev/GrammarEngine/tree/master/src/bin-windows64 или
собрать ее самостоятельно из исходных текстов, которые лежат здесь https://github.com/Koziev/GrammarEngine/tree/master/src/dictionary.src

## Запуск

Утилита запускается в консоли MS Windows. Необходимо указать путь к собранной
русской словарной базе, путь к исходному файлу с фактами и путь к папке с результатами
работы:

```
GenerateQAFromParsing.exe -dict e:\MVoice\lem\bin-windows64\dictionary.xml -input e:\polygon\paraphrasing\data\facts4.txt -output f:\tmp
```

