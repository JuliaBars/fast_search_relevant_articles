import sys

import streamlit as st

import settings
import utils
from config import df, embedder_model, index_IP, index_l2

st.set_page_config(
    page_title="Поиск статей Gazeta.ru похожих на ваш запрос",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="auto",
)

st.header('Поиск статей Gazeta.ru похожих на ваш запрос', divider='rainbow')
st.sidebar.header('Настройки')
n_articles: int = int(st.sidebar.slider(
    'Выберите количество статей в ответе', 2, 5, 10))

source_radio = st.sidebar.radio(
    'Попробуйте разные алгоритмы поиска', settings.INDEX_LIST
    )


if __name__ == '__main__':
    try:
        query = st.text_area("Введите запрос и нажмите Подобрать")
        if source_radio == settings.INDEXFLATL2:
            index = index_l2
        if source_radio == settings.INDEXFLATIP:
            index = index_IP
        if st.button("Подобрать"):
            result = utils.top_articles(
                index, query, embedder_model, df, n_articles
                )
            st.write('Похожие статьи:', result)
    except Exception as err:
        utils.logger.error('Произошла ошибка: ', err)
        sys.exit(0)
