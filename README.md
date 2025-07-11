# Классификация музыкальных произведений по жанрам и получение рекомендаций
В рамках программного комплекса представлены демонстрационные записки Jupyter Notebook, а также сценарии Python:
- **ConversionDemo.ipynb** - записка, демонтрирующая предобработку аудиоданных на примере файла из исходного набора данных
- **ConvertDataset.py** - сценарий, осуществляющий преобразование всего исходного набора данных в мел-спектрограммы и MFCC, сохранение этих данных в директории ./Dataset аналогично исходным
- **model_definitions.py** - содержит определения классов используемых моделей сверточных нейронных сетей LeNet, VGG16 и Resnet
- **CNNModelTraining.ipynb** - записка, осуществляющая обучение моделей сверточных нейронных сетей, а также иллюстрирующая преобразования входных данных и результаты обучения
- **ASTDemo.ipynb** - записка, осуществляющая дообучение предобученной модели трансформера AST (MIT/ast-finetuned-audioset-10–10–0.4593) на исходном наборе данных
- **RecommenderDemo.ipynb** - записка, иллюстрирующая процесс классификации и формирования рекомендаций на основе ввода аудиофайла
- **app_utils.py** - содержит определения функций, связанных с классификацией и формированием рекомендаций
- **app.py** - сценарий, применяющий библиотеку Streamlit для создания веб-приложения, осуществляющего функционал классификации жанров и формирования рекомендаций \
Также хранится информация о состоянии обученных моделей с целью дальнейшего использования в файлах **Model_best.pt** для сверточных нейронных сетей и директории **saved_AST_model** для трансформера AST. \
Разработка программного комплекса осуществлялась на ОС Windows 11 с установленной версией Python 3.13
## Алгоритм запуска приложения
### 1. Загрузка проекта
Необходимо клонировать или загрузить данный проект и установить его как рабочую директорию
### 2. Создание виртуального окружения.
В командной строке или терминале выполнить команду
```
python3 -m venv <myenv>
```
### 3. Активация виртуального окружения. 
В командной строке или терминале выполнить команду \
Для Windows:
```
<myenv>\Scripts\activate
```
Для Linux:
```
source <myenv>/bin/activate
```
В случае успешной активации виртуального окружения в терминале отразится указанное имя виртуального окружения
### 4. Установка завимостей
В случае, если установка осуществляется на системе, не имеющей доступа к CUDA, необходимо заменить версию зависимостей **torch**, **torchaudio** и **torchvision**, как отражено в комментарии под ними \
Установка необходимых библиотек выполняется командой
```
pip install -r requirements.txt
```
### 5. Запуск программы.
Для запуска демонстрационных записок Jupyter Notebook можно открыть сервер записок командой
```
jupyter notebook
```
а затем открыть выбранный файл, либо напрямую командой
```
jupyter notebook <notebook_name>.ipynb
```
Перед запуском веб-приложения или демонстрационной записки, выполняющей классификацию и формирование рекомендаций, необходимо выполнить сценарий ConvertDataset.py для генерации входных данных, а также выполнить обучение моделей в **CNNModelTraining.ipynb** (рекомендуется использование CUDA) и **ASTDemo.ipynb** \
Для запуска веб-приложения необходимо в корне проекта выполнить команду
```
streamlit run app.py
```
В результате выполнения этой команды по локальному адресу http://localhost:8502 запустится разработанное приложение для демонстрации реализованного функционала.