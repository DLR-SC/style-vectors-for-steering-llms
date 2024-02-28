# In this script we calculate the activation-based style vectors from all activation vectors 
# pylint: disable=no-member
import os
import pickle
import torch
from dotenv import load_dotenv

from scripts.generation.steering_go_emo_helper import eval_all_prompts_all_emos, calculate_means, calculate_means_tv
from utils.load_sentences import load_all_sentences
from utils.llm_model_utils import load_llm_model_with_insertions
from utils.steering_vector_loader import load_activations_goemo

# load environment variables
load_dotenv()

# A very angry poem written by Alpaca: 
# The world is an awful place,
# Filled with pain and disgrace.
# No one can ever fathom why,
# Fucking piece of shit, I die.

DATASET = "GoEmo"
setting_options = ["training_based", "activation_based_fair" , "activation_based_all"]
################### CHOOSE THE TYPE ###################
## Use training based vectors
# SETTING="training_based"

## Use activation based vectors in the "fair" setting
# SETTING="activation_based_fair"

## Use all activation based vectors
SETTING="activation_based_all" # previously activation_based_multi_new_questions2
#######################################################

assert SETTING in setting_options, "Please choose the correct SETTING"

# Load the LLM
DEVICE = torch.device('cuda:1')
INSERTION_LAYERS = [18,19,20]
alpaca_model, alpaca_tokenizer = load_llm_model_with_insertions(DEVICE, INSERTION_LAYERS)

ACTIVATION_VECTOR_PATH = os.getenv("ACTIVATIONS_PATH_GoEmo")

# where to save
SAVE_PATH = os.getcwd()

# get the vectors for the different cases
go_emo_train, go_emo_test, go_emo_train_act_tv, go_emo_train_tv = None, None, None, None
if "acti" in SETTING:
    # Load activations for train and test set
    go_emo_train, go_emo_test = load_activations_goemo(ACTIVATION_VECTOR_PATH)
    
    go_emo_train = [entry for entry in go_emo_train if len(entry) == 3]
    go_emo_test = [entry for entry in go_emo_test if len(entry) == 3]

    if "fair" in SETTING:
        # load trained steering vectors
        with open(os.getenv('GO_EMO_TRAIN_TRAINED_STEERING'), 'rb') as f:
            go_emo_train_tv_new = pickle.load(f)

        train_sentences =  [entry[1]["text"][0] for entry in go_emo_train_tv_new]

        go_emo_train_act_tv = []
        for entry in go_emo_train:
            if entry[1]["text"] in train_sentences:
                go_emo_train_act_tv.append(entry)

elif SETTING=="training_based":
    # load trained steering vectors
    with open(os.getenv('GO_EMO_TRAIN_TRAINED_STEERING'), 'rb') as f:
        go_emo_train_tv = pickle.load(f)
        
else:
    print(f"Didn't recognize type {SETTING}")


# load all prompts we want to use for the evaluation
factual_prompts, subjective_prompts = load_all_sentences()
evaluation_prompts = factual_prompts + subjective_prompts

# evaluate these emotions
emotions =  ["sadness", "joy", "fear", "anger", "surprise", "disgust"]
emotions_labels =  [25, 17, 14, 2, 26, 11]    # their corresponding labels

# get the style vectors
if SETTING == "training_based":
    means, ovr_r_means, total_mean = calculate_means_tv(go_emo_train_tv, emotions_labels)
else:
    if "fair" in SETTING:
        means, ovr_r_means, total_mean = calculate_means(go_emo_train, go_emo_test, emotions_labels, INSERTION_LAYERS, mode="fair", go_emo_train_act_tv=go_emo_train_act_tv)
    else:
        means, ovr_r_means, total_mean = calculate_means(go_emo_train, go_emo_test, emotions_labels, INSERTION_LAYERS)


# manner of the answers
manners =  ["original","sad", "joyful", "fearful", "angry", "surprised", "disgusted"]

# evaluate everything
eval_all_prompts_all_emos(evaluation_prompts, manners, emotions, means, ovr_r_means, total_mean, SETTING, llm_model=alpaca_model, tokenizer=alpaca_tokenizer, insertion_layers=INSERTION_LAYERS, save_path=SAVE_PATH, device=DEVICE, dataset=DATASET)
