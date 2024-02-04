EMB_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
PATH_TO_DATA = 'data/gazeta_train.jsonl'
PATH_TO_SAVE = 'data/embedded_data.pkl'
COLUMN_WITH_EMB = 'embeddings'
COLUMN_WITH_TEXT = 'summary'
PATHS_TO_PKL = [
    'data/embedded_data_1.pkl',
    'data/embedded_data_2.pkl',
    'data/embedded_data_3.pkl',
    'data/embedded_data_4.pkl'
    ]
INDEXFLATL2 = 'IndexFlatL2'
INDEXFLATIP = 'IndexFlatIP'

INDEX_LIST = [INDEXFLATIP, INDEXFLATL2]
