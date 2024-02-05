import settings
import utils


def config_project(
        embedder_name: str,
        column_with_embed: str,
        column_with_text: str,
        path_to_data: str,
        paths_to_pkl: list[str],
        pre_embeds: bool = True,
        ) -> tuple[
            utils.faiss.IndexFlatL2,
            utils.faiss.IndexFlatIP,
            utils.SentenceTransformer,
            utils.pd.DataFrame | None,
            ]:
    embedder = utils.Embedding()
    embedder_model = embedder.load_model(embedder_name)
    if pre_embeds:
        df = embedder.load_emdedds_from_file(
            utils.read_from_pkl, paths_to_pkl
            )
    else:
        df = utils.prepare_df(path_to_data)
        embedder.embeddings_to_df(
            df, embedder_model, column_with_text, column_with_embed
            )
    index_l2 = utils.SearchVectors().get_index(
        df, column_with_embed, settings.INDEXFLATL2
        )
    index_IP = utils.SearchVectors().get_index(
        df, column_with_embed, settings.INDEXFLATIP
        )
    return index_l2, index_IP, embedder_model, df


index_l2, index_IP, embedder_model, df = config_project(
    settings.EMB_MODEL,
    settings.COLUMN_WITH_EMB,
    settings.COLUMN_WITH_TEXT,
    settings.PATH_TO_DATA,
    settings.PATHS_TO_PKL,
    )
