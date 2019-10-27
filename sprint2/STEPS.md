# Инструкция по выполнению

## Шаг 1. Откройте таблицу и изучите общую информацию о данных

Путь к файлу: `/datasets/data.csv`

## Шаг 2. Предобработка данных

1. определите и заполните пропущенные значения:
2. опишите, какие пропущенные значения вы обнаружили;
3. приведите возможные причины появления пропусков в данных;
4. объясните, по какому принципу заполнены пропуски;
5. замените вещественный тип данных на целочисленный:
6. поясните, как выбирали метод для изменения типа данных;
7. удалите дубликаты:
8. поясните, как выбирали метод для поиска и удаления дубликатов в данных;
9. приведите возможные причины появления дубликатов;
10. выделите леммы в значениях столбца с целями получения кредита:
11. опишите, как вы проводили лемматизацию целей кредита;
12. категоризируйте данные:
13. перечислите, какие «словари» вы выделили для этого набора данных, и объясните, почему.

## Шаг 3. Ответьте на вопросы

* Есть ли зависимость между наличием детей и возвратом кредита в срок?
* Есть ли зависимость между семейным положением и возвратом кредита в срок?
* Есть ли зависимость между уровнем дохода и возвратом кредита в срок?
* Как разные цели кредита влияют на его возврат в срок?

Ответы сопроводите интерпретацией — поясните, о чём именно говорит полученный вами результат.

## Шаг 4. Напишите общий вывод

Оформление: Задание выполните в Jupyter Notebook. Программный код заполните в ячейках типа code,
текстовые пояснения — в ячейках типа markdown. Примените форматирование и заголовки.

## Описание данных

* `children` — количество детей в семье
* `days_employed` — общий трудовой стаж в днях
* `dob_years` — возраст клиента в годах
* `education` — уровень образования клиента
* `education_id` — идентификатор уровня образования
* `family_status` — семейное положение
* `family_status_id` — идентификатор семейного положения
* `gender` — пол клиента
* `income_type` — тип занятости
* `debt` — имел ли задолженность по возврату кредитов
* `total_income` — ежемесячный доход
* `purpose` — цель получения кредита

## Как будут проверять мой проект

На что обращают внимание наставники при проверке проектов:

* Как вы описываете найденные в данных проблемы?
* Какие методы замены типов данных, обработки пропусков и дубликатов применяете?
* Умеете лемматизировать?
* Категоризируете данные? Почему именно таким образом?
* Выводите ли финальные данные в сводных таблицах?
* Применяете ли конструкцию try-except для обработки потенциальных ошибок?
* Соблюдаете ли структуру проекта и поддерживаете аккуратность кода?
* Какие выводы делаете?
* Оставляете ли комментарии к шагам?