def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64 ]),
        }


all_configs = {}
_SHAKESPEARE_CONFIG1_ = "shakespeare_classifier1"
_SHAKESPEARE_CONFIG2_ = "shakespeare_classifier2"
_SHAKESPEARE_CONFIG3_ = "shakespeare_classifier3"
_SHAKESPEARE_CONFIG4_ = "shakespeare_classifier4"
_SHAKESPEARE_CONFIG5_ = "shakespeare_classifier5"
_SHAKESPEARE_CONFIG6_ = "shakespeare_classifier6"
_SHAKESPEARE_CONFIG7_ = "shakespeare_classifier_model"
_SHAKESPEARE_HP_SEARCH_ = "shakespeare_classifier_hp"

all_configs[_SHAKESPEARE_CONFIG1_] = {
    "dataset_name": "notaphoenix/shakespeare_dataset",
    "undersample": False,
    "pretrained_model_name": "distilbert-base-uncased", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir": _SHAKESPEARE_CONFIG1_,
    "learning_rate": 5e-6,
    "per_device_train_batch_size" :16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 20,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": True,
    "search_hp": False,
    "optuna_hp_func": None
}


all_configs[_SHAKESPEARE_HP_SEARCH_] = {
    "dataset_name": "notaphoenix/shakespeare_dataset",
    "undersample": False,
    "pretrained_model_name": "distilbert-base-uncased", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir": _SHAKESPEARE_HP_SEARCH_,
    "learning_rate": 5e-6,
    "per_device_train_batch_size" :16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 20,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": False,
    "search_hp": True,
    "optuna_hp_func": optuna_hp_space
}

#microsoft/deberta-v3-base


all_configs[_SHAKESPEARE_CONFIG2_] = {
    "dataset_name": "notaphoenix/shakespeare_dataset",
    "undersample": False,
    "pretrained_model_name": "microsoft/deberta-v3-base", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir": _SHAKESPEARE_CONFIG2_,
    "learning_rate": 5e-6,
    "per_device_train_batch_size" :32,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 5,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": True,
    "search_hp": False,
    "optuna_hp_func": None
}

all_configs[_SHAKESPEARE_CONFIG3_] = {
    "dataset_name": "notaphoenix/shakespeare_dataset",
    "undersample": False,
    "pretrained_model_name": "microsoft/deberta-v3-base", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir": _SHAKESPEARE_CONFIG3_,
    "learning_rate": 5e-6,
    "per_device_train_batch_size" :16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 5,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": True,
    "search_hp": False,
    "optuna_hp_func": None
}


# Big batch or big LR
all_configs[_SHAKESPEARE_CONFIG4_] = {
    "dataset_name": "notaphoenix/shakespeare_dataset",
    "undersample": False,
    "pretrained_model_name": "microsoft/deberta-v3-base", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir": _SHAKESPEARE_CONFIG2_,
    "learning_rate": 4e-6,
    "per_device_train_batch_size" :32,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 5,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": True,
    "search_hp": False,
    "optuna_hp_func": None
}

all_configs[_SHAKESPEARE_CONFIG5_] = {
    "dataset_name": "notaphoenix/shakespeare_dataset",
    "undersample": False,
    "pretrained_model_name": "microsoft/deberta-v3-base", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir": _SHAKESPEARE_CONFIG2_,
    "learning_rate": 5e-7,
    "per_device_train_batch_size" :16,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 50,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": True,
    "search_hp": False,
    "optuna_hp_func": None
}

all_configs[_SHAKESPEARE_CONFIG6_] = {
    "dataset_name": "notaphoenix/shakespeare_dataset",
    "undersample": False,
    "pretrained_model_name": "microsoft/deberta-v3-base", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir": _SHAKESPEARE_CONFIG6_,
    "learning_rate": 5e-6,
    "per_device_train_batch_size" :16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 5,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": True,
    "search_hp": True,
    "optuna_hp_func": optuna_hp_space
}

all_configs[_SHAKESPEARE_CONFIG7_] = {
    "dataset_name": "notaphoenix/shakespeare_dataset",
    "undersample": False,
    "pretrained_model_name": "distilbert-base-uncased", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir": _SHAKESPEARE_CONFIG7_,
    "learning_rate": 2e-6,
    "per_device_train_batch_size" :16,
    "per_device_eval_batch_size": 64,
    "num_train_epochs": 12,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": True,
    "search_hp": False,
    "hub_private_repo": False,
    "optuna_hp_func": None
}