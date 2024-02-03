### Cистема обратного поиска (reverse text search) для кратких описаний статей с ресурса Gazeta.ru


---
Система позволяет искать самые похожие на введённый пользователем запрос статьи.

---

**Особенности проекта:**

1. Использован датасет со статьями Gazeta.ru [link](https://huggingface.co/datasets/IlyaGusev/gazeta) (61K статей)

2. Для получения эмбедингов рассмотрены 3 модели:
- paraphrase-multilingual-mpnet-base-v2 (по ТЗ)
- mstsb-paraphrase-multilingual-mpnet-base-v2
- LaBSE (Language-agnostic BERT Sentence Embedding)

3. Для быстрого поиска по эмбедингам использована библиотека FAISS, рассмотрена работа двух индексов, их можно попробовать через GUI:
- IndexFlatIP (косинусное расстояние по ТЗ)
- IndexFlatL2 (евклидово расстояние)

4. Рассмотрена возможность использование большой языковой модели по API для переранжирования текстов по релевантности после поиска по векторным представлениям. 

- Использования внешнего API - дорого.
- Локально развернуть - невозможно из-за недостатка мощностей на бесплатном сервере Streamlit.

5. GUI разработан на фреймворке Streamlit.

6. Добавлена возможность выбирать количество статей в выдаче.

7. Выдача по двум индексам сохранена в файл result.txt


Локально запустить проект можно:
- используя Docker
```
docker build -t articles_search .
docker run --name articles  --rm -it -p 8501:8501 articles_search
http://localhost:8501/
```
- используя виртуальное окружение
```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
http://localhost:8501/
```

### Проблемы возникшие при разработке:

- очень краткая документации FAISS, чтобы разобраться как все работает, приходилось смотреть исходный код, wiki и SO
- Довольно медленная работа модели для эмбедингов без параллельности, проблему удалось решить, сохранив файл pkl с рассчитанными эмбедингами
- Проект задеплоен на бесплатных серверах Streamlit (к сожалению, иногда они отключаются)
- На гит нельзя отправить файлы больше 100МB, пришлось разрезать файл c эмбедингами на части, а файл с исходных датасеом добавила в описание ссылкой