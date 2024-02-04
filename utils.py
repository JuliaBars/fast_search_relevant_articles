from typing import Callable
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import sys
from typing import Any
from loguru import logger


def config_logger(logger):
    logger.remove()
    logger.add(
        sys.stderr, format="[{time}] [<level>{level}</level>] {message}"
        )
    logger.level("INFO", color="<green>")
    logger.level("ERROR", color="<red>")


config_logger(logger)


class Embedding:
    """Создание эмбедингов для текста."""

    def load_model(self, model_name) -> SentenceTransformer:
        """Загрузка модели для эмбедингов"""
        try:
            model = SentenceTransformer(model_name)
            logger.info("Модель embedder загружена успешно")
            return model
        except Exception as err:
            logger.error('Не удалось загрузить модель embedder: ', err)
            sys.exit(0)

    def embeddings_to_df(
            self,
            df: pd.DataFrame,
            model: SentenceTransformer,
            column_with_text: str,
            column_with_emb: str,
            ) -> None:
        """Создаем эмбеддинги для датафрейма in-place."""
        try:
            embeddings = model.encode(df[column_with_text])
            logger.info("Эмбединги текстов для df созданы успешно")
        except Exception as err:
            logger.error('Не удалось создать эмбединги: ', err)
            sys.exit(0)
        df[column_with_emb] = embeddings.tolist()
        df[column_with_emb] = df[column_with_emb].apply(lambda x: np.array(x))

    def one_embedding(self, model: SentenceTransformer, text: str) -> Any:
        """Создаем эмбединги для 1 текста"""
        try:
            embeddings = model.encode(text)
            logger.info("Эмбединг одиночного текста создан успешно")
            return embeddings
        except Exception as err:
            logger.error('Не удалось создать эмбединги: ', err)
            sys.exit(0)

    def load_emdedds_from_file(
            self, func: Callable,
            paths_to_files: list[str],
            ) -> pd.DataFrame:
        return func(paths_to_files)


class SearchVectors:
    def get_index(self, df, column: str, index_method) -> Any:
        vectors = np.stack(df[column].values, axis=0)
        try:
            if index_method == 'IndexFlatL2':
                indexes = faiss.IndexFlatL2(vectors.shape[1])
            if index_method == 'IndexFlatIP':
                indexes = faiss.IndexFlatIP(vectors.shape[1])
            indexes.add(vectors)
            logger.info("Поисковые индексы созданы успешно")
            return indexes
        except Exception as err:
            logger.error('Не удалось создать поисковые индексы: ', err)
            sys.exit(0)


def save_to_pkl(df: pd.DataFrame, path_to_save: str) -> None:
    """Сохранение dataframe в pkl формате."""
    try:
        df.to_pickle(path_to_save)
        logger.info("Файл pkl успешно сохранен")
    except Exception as err:
        logger.error('Не удалось сохранить файл: ', err)


def read_from_pkl(paths_to_files: list[str]) -> pd.DataFrame | None:
    try:
        df = pd.DataFrame()
        for path in paths_to_files:
            df1 = pd.read_pickle(path)
            df = pd.concat([df, df1], axis=0)
        logger.info('Создан DataFrame из файла pkl')
        return df
    except Exception as err:
        logger.error('Загрузка датасета из pkl не удалась: ', err)


def prepare_df(path_to_file: str) -> pd.DataFrame | None:
    """Загружаем датасет, удаляем ненужные столбцы."""
    try:
        df = pd.read_json(path_to_file, lines=True)
        df.drop(columns=['text', 'date', 'title'], inplace=True)
        logger.info('Датасет успешно загружен из jsonl')
        return df
    except Exception as err:
        logger.error('Не удалось загрузить датасет из jsonl: ', err)
        sys.exit(0)


def top_articles(
        indexes, query, embedder, df, n: int = 10
        ) -> np.ndarray | None:
    """Возвращает n самых похожих текстов."""
    embedded_query = embedder.encode(query)
    logger.info('Система готова к поиску по индексам')
    try:
        _, idx = indexes.search(np.array(embedded_query).reshape((1, 768)), n)
        logger.info('Поиск по индексам успешно завершен')
    except Exception as err:
        logger.error('Поиск по индексам не завершен: ', err)
        sys.exit(0)
    result = df.iloc[idx[0]][['summary', 'url']].reset_index(drop=True)
    result.index = pd.RangeIndex(start=1, stop=len(result) + 1)
    result.rename(
        columns={'summary': 'Краткое содержание', 'url': 'Ссылка'},
        inplace=True
        )
    return result
