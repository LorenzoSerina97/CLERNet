RANDOM_SEED = 3993

LOSSES_SHORT_NAMES = {
    "binary_crossentropy": "bce",
    "binary_focal_crossentropy": "bfce",
    "rmse": "rmse",
    "rmse_ol": "rmse_ol",
    "rmse_op": "rmse_op",
    "rmse_hmlp": "rmse_hmlp",
    "rmse_mlp": "rmse_mlp",
    "bce_ol": "bce_ol",
    "bce_op": "bce_op",
    "bce_hmlp": "bce_hmlp",
    "bce_mlp": "bce_mlp",
    "bfce_ol": "bfce_ol",
    "bfce_op": "bfce_op",
    "bfce_hmlp": "bfce_hmlp",
    "bfce_mlp": "bfce_mlp",
}

MAX_PLAN_DIMS = {
    "blocksworld": 75,
    "depots": 50,
    "driverlog": 70,
    "logistics": 50,
    "satellite": 40,
    "zenotravel": 40,
}


class DEFAULTS:
    MAX_PLAN_DIM = 50
    MIN_PLAN_PERC = 0.3
    MAX_PLAN_PERC = 1.0
    TRAIN_PERCENTAGE = 0.7
    BATCH_SIZE = 64
    EPOCHS = 50
    LOSS_FUNCTION = "binary_crossentropy"
    RUN_OPTUNA_FLAG = False
    OPTUNA_TRIALS = 20
    WARMUP_EPOCHS = 10


class ERRORS:
    MSG_ERROR_BOTH_ATTENTION_TIME_DISTRIB = (
        "Error in the selected parameters.\n"
        "Cannot use both Attention Mechanism and Time Distributed layers."
    )
    MSG_ERROR_LOAD_PARAMS = "Error while loading the parameters"
    MSG_ERROR_LOAD_PLANS = f"Error while loading the plans"
    STD_ERROR_LOAD_FILE = "Error while loading {0}"
    STD_LOAD_FILE_OK = "{0} loaded from {1}"
    STD_FILE_NOT_SAVED = "{0} was not saved"
    MSG_ERROR_LOAD_MODEL = "Error while loading the model {0} from {1}."
    MSG_LOAD_MODEL_OK = "The model {0} was loaded from {1}"
    MSG_ERROR_CREATE_PREDICTIONS = "Could not create the predictions file(s)"


class CREATE_TRAIN_TEST:
    TRAIN_PLANS_NUMBER = "Total train plans: {0}"
    TEST_PLANS_NUMBER = "Total test plans: {0}"
    VALIDATION_PLANS_NUMBER = "Total validation plans: {0}"


class FILENAMES:
    STATS_FILENAME = "stats.txt"
    TRAIN_PLANS_FILENAME = "train_plans"
    VALIDATION_PLANS_FILENAME = "val_plans"
    TEST_PLANS_FILENAME = "test_plans"
    TRAIN_PLANS_FILENAME_SIMPLE_JSON = "train_plans_simple.json"
    VALIDATION_PLANS_FILENAME_SIMPLE_JSON = "val_plans_simple.json"
    COMPACT_TRAIN_PLANS_FILENAME_SIMPLE_JSON = "compact_train_plans_simple.json"
    COMPACT_VALIDATION_PLANS_FILENAME_SIMPLE_JSON = "compact_val_lpg_plans_simple.json"
    TRAIN_PLANS_FILENAME_ONLINE_JSON = "train_plans_online.json"
    VALIDATION_PLANS_FILENAME_ONLINE_JSON = "val_plans_online.json"
    TEST_PLANS_FILENAME_SIMPLE_JSON = "test_plans_simple.json"
    PLOT_ACTIONS_FILENAME = "actions_frequency.png"
    PLOT_GOALS_FILENAME = "goals_frequency.png"
    PLOT_LENGTH_FILENAME = "plans_length.png"
    PLANS_FILENAME = "plans"
    ACTION_DICT_FILENAME = "dizionario"
    ACTION_DICT_FILENAME_W2V = "dizionario_w2v"
    GOALS_DICT_FILENAME = "dizionario_goal"
    NETWORK_PLOTS_FOLDER = "plots"
    PLANS_FOLDER = "plans_max-plan-dim={0}_train_percentage={1}"
    PARAMS_TEMPLATE_FILENAME = "params_template"


class CREATE_DATASET:
    GOALS_TABLE_TITLE = "Goals frequency"
    ACTIONS_TABLE_TITLE = "Actions frequency"
    PLANS_TABLE_TITLE = "Plans length"
    PLANS_NUMBER = "Total plans : {0}"
    GOALS_NUMBER = "Total goals: {0}"
    ACTIONS_NUMBER = "Total actions: {0}"
    TABLE_HEADERS = ["MIN", "Q1", "Q2", "Q3", "MAX"]
    NBINS = 50


class HELPS:
    DATASET_TYPE = "Type of created problems."
    DOMAIN = "PDDL domain of the problems."
    STARTING_DIR_SRC = "Folder that contains the datasets folder."
    PYTHON_FILE = "Path to the python bin file."
    SOLVER_SOLUTIONS = "Name of the solver used for generating the solutions."
    LPG_SOLVER_FILE_SRC = "Path to LPG solver."
    XML_FOLDER_OUT = "Folder where to save the created XML files."
    SOLUTIONS_FOLDER_SRC = "Folder that contains the SOL files."
    PROCESSORS_NUMBER = "Number of processors for parallel execution."
    CPU_TIME = "Maximum CPU time for each process."
    SOLUTIONS_NUMBER = "Number of solutions created for each problem (LPG only)."
    SOLUTIONS_FOLDER_OUT = "Folder where to save the SOL files."
    SOLVER_FILE_SRC = "Path to the solver file."
    PDDL_FILE_FOLDER_SRC = "Folder that contains the pddl problems files."
    PARAMETERS_NUMBER = "Number of parameters templates to generate."
    PARAMS_TEMPLATE_DIR_OUT = "Folder where to store the created template file(s)."
    SAVE_STATS_FLAG = "Flag for creating the file stats.txt that saves the stats in the plots directory."
    PRED_DIR_SRC = "Folder that contains the prediction file(s)."
    PRED_DIR_OUT = "Folder where to save the predictions."
    PLAN_PERCENTAGE = "Percentage of actions per plan."
    TEST_PLANS_DIR_SRC = "Folder that contains the test plans."
    MODEL_SRC = "Path of the model file to use."
    MODEL_NAME = "Name of your model; it will also be the name of the Optuna Study."
    DB_DIR = "Folder that contains the database where to store the Optuna Study."
    TRIALS = "Number of Optuna trials."
    INCREMENTAL_TESTS_FLAG = "Flag for incremental tests."
    PLANS_FOLDER_SRC = "Folder that contains the train, test and validation plans."
    MIN_PLAN_PERCENTAGE = "Minimum percentage of actions per plan considered."
    EPOCHS = "Number of epochs for training the network."
    DICT_FOLDER_SRC = "Folder that contains the actions and goals dictionaries."
    BATCH_SIZE = "Size of each batch"
    MAX_PLAN_PERCENTAGE = "Maximum percentage of action per plan considered."
    MODEL_DIR_OUT = "Folder where to save the model directory."
    NETWORK_PARAMETERS_SRC = "Path to the network parameters file."
    CREATE_IF_NOT_EXISTS = "It's created if it does not exists."
    XML_FOLDER_SRC = "Folder that contains the XMLs files."
    PLANS_AND_DICT_FOLDER_OUT = "Folder where to store plans and dictionaries file."
    ONEHOT_FLAG = "Flag that applies the one-hot representation for the goals."
    PLOTS_FOLDER_OUT = "Folder where to save the plots."
    TRAIN_TEST_VAL_FOLDER_OUT = (
        "Folder where to save the train, test and validation files."
    )
    PLANS_AND_DICT_FOLDER_SRC = (
        "Folder that contains the plans and dictionaries pickles."
    )
    MAX_PLAN_LENGTH = "Maximum plan length accepted."
    TRAIN_PERCENTAGE = "Percentage of plans used to create the training set."
    NO_VAL_FLAG = "Flag used not to create the validation set"
    PARAMS_TEMPLATE = "Path to the parameters template file (JSON)."
    LOSS_FUNCTION = "Loss function used for training the network."
    GOAL_REC_TEST_PLANS_DIR = "Folder that contains the goal recognition test plans."
    RUN_OPTUNA_FLAG = "Flag for running Optuna."
    DRY_RUN_FLAG = "Flag for running a dry run."
    DOMAIN_NAME = "The planning domain name."


class KEYS:
    MODEL_NAME = "MODEL_NAME"
    STUDY = "STUDY"
    TARGET_DIR = "TARGET_DIR"
    MAX_PLAN_DIM = "MAX_PLAN_DIM"
    MIN_PLAN_PERC = "MIN_PLAN_PERC"
    MAX_PLAN_PERC = "MAX_PLAN_PERC"
    READ_PLANS_DIR = "READ_PLANS_DIR"
    BATCH_SIZE = "BATCH_SIZE"
    EPOCHS = "EPOCHS"
    GOALS_DICT = "GOALS_DICT"
    ACTION_DICT = "ACTION_DICT"
    MODEL_DIR = "MODEL_DIR"
    PARAMS = "PARAMS"
    PLANS = "PLANS"
    LOSS_FUNCTION = "LOSS_FUNCTION"
    READ_DICT_DIR = "READ_DICT_DIR"
    GOAL_REC_RESULTS_DIR = "GOAL_REC_RESULTS_DIR"


class PARAMS_GEN:
    DEFAULT_MODEL_NAME = "MODEL_NAME_TO_CHANGE"


class GENERATE_FILES:
    XML = "xml"
    SOL = "sol"
    LPG = "LPG"
    FF = "FF"
    DOMAIN_FILE = "domain.pddl"
    HOME = "$HOME"
