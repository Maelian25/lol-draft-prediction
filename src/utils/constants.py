# GENERAL CONSTANTS
ROLE_MAP = {"TOP": 1, "JUNGLE": 2, "MIDDLE": 3, "BOTTOM": 4, "SUPPORT": 5}
SAVE_AFTER_ITERATION = 1000
REGIONS = dict({"EUROPE": "EU", "KOREA": "KR", "AMERICA": "US"})
ELOS = dict({"challenger": 125, "grandmaster": 250, "master": 500})

# FOLDERS
DATA_REPRESENTATION_FOLDER = "./data/representation/"
MODELS_PARAMETER_FOLDER = "./data/models_parameter/"
MATRICES_FOLDER = "./data/matrices/"
DATASETS_FOLDER = "./data/datasets/"

# FILENAMES
BT_MODEL = "BTFeature_param.pth"
DRAFT_STATES_PARQUET = "game_states.parquet"
DRAFT_STATES_CSV = "game_states.csv"
CHAMP_EMBEDS = "champion_embeddings.parquet"
COUNT_MAT = "counter_matrix.parquet"
SYN_MAT = "synergy_matrix.parquet"
