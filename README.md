# jet-project

Репозиторий команды 3 курса МААД МКН СПбГУ по проекту от "Инфосистемы Джет" по распознаванию серийных номеров на металлических заготовках.

## Версии

Принципиальным различием между версиями `master` и `experimental` является то, что решение `master` старается обрабатывать файлы по одному и не держит множество изображений в оперативной памяти, выгружая их на диск при каждой удобной возможности, в то время как `experimental` считывает файлы один раз в память и работает не с ранее выгруженными на диск данными, а с теми же, что были считаны в самом начале.

Также в ветке `experimental` лежит демонстрация работы пайплайна, написанная с помощью Streamlit.

## Запуск пайплайна

Запуск осуществляется с помощью скрипта `Pipeline/pipeline.py`, в который через параметр `--source` нужно передать директорию с изображениями на обработку. Результаты появятся в директории, из которой был запущен скрипт. Также вызовом команды
```
python Pipeline/pipeline.py --help
```
можно ознакомиться с дополнительными параметрами, влияющими на вывод скрипта.

## Пререквизиты

Для успешного запуска пайплайна необходимо активировать виртуальную среду Python и установить в нее необходимые модули (см. файл `Pipeline/requirements.txt`) с помощью команды
```
pip install -r Pipeline/requirements.txt
```

Также для запуска пайплайна из ветки `master` необходимо склонировать [YOLOv5](https://github.com/ultralytics/yolov5) в корень репозитория и доустановить модули из файла `requirements.txt` для YOLOv5.

## Замечания для команды
Файлы и папки с рабочим кодом (`.ipynb`, `.py` и так далее) называются с большой буквы, остальные - с маленькой (исключением являются файлы пайплайна). В названиях файлов вместо пробелов используются нижние подчеркивания.
Если файл пишет один участник команды и ему важно, чтобы другими он не изменялся, в начале файла нужно ставить префикс, составленный из инициалов участника команды.

`git add --all` является запретным заклинанием и его использование поощрению не подлежит.
