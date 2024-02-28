# Steering for YELP
# pylint: disable=no-member
import os
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from scripts.generation.steering_yelp_helper import run_llm_steering
from utils.llm_model_utils import load_llm_model_with_insertions

# load environment variables
load_dotenv()

from utils.steering_vector_loader import load_trained_vectors_yelp, load_activations_yelp
from utils.steering_layer import SteeringLayer
from utils.load_sentences import load_all_sentences

# define the path
SAVE_PATH=os.getcwd()

DEVICE = torch.device('cuda:0')
DATASET="yelp"

################### CHOOSE THE TYPE ###################
## Use training based vectors
# SETTING="training_based" # previously trained_vector_based

## Use all activation based vectors
SETTING="activation_based_all" # previously activation_based_multi_new_questions2
#######################################################

INSERTION_LAYERS = [18,19,20]
alpaca_model, alpaca_tokenizer = load_llm_model_with_insertions(DEVICE, INSERTION_LAYERS)

if SETTING == "training_based":
    # load the trained vectors
    VECTOR_PATH = os.getenv("TRAINED_VECTORS_PATH_Yelp")
    steering_vectors = load_trained_vectors_yelp(VECTOR_PATH)

elif "acti" in SETTING:
    # load the activation vectors
    VECTOR_PATH = os.getenv("ACTIVATIONS_PATH_YELP")
    steering_vectors = load_activations_yelp(VECTOR_PATH, DATASET)

positive = [sv for sv in steering_vectors if sv[-1] == 1]
negative = [sv for sv in steering_vectors if sv[-1] == 0]

################################b########################################
pos_actis = []
neg_actis = []
for sv in tqdm(positive, desc="positives"):
    input_tokens = alpaca_tokenizer(sv[1].replace('\n',''), return_tensors="pt").to(DEVICE)
    gen_text = alpaca_model.forward(input_tokens.input_ids, output_hidden_states=True)
    # pos_actis.append(gen_text[2][15].detach().cpu().numpy()[0])
    pos_actis.append([gen_text[2][18][0][-1].detach().cpu().numpy(),gen_text[2][19][0][-1].detach().cpu().numpy(),gen_text[2][20][0][-1].detach().cpu().numpy()])
    # pos_actis.append([gen_text[2][15][0][-1].detach().cpu().numpy(),gen_text[2][16][0][-1].detach().cpu().numpy(),gen_text[2][17][0][-1].detach().cpu().numpy()])

for sv in tqdm(negative, desc="negatives"):
    input_tokens = alpaca_tokenizer(sv[1].replace('\n',''), return_tensors="pt").to(DEVICE)
    gen_text = alpaca_model.forward(input_tokens.input_ids, output_hidden_states=True)
    # neg_actis.append(gen_text[2][15].detach().cpu().numpy()[0])
    neg_actis.append([gen_text[2][18][0][-1].detach().cpu().numpy(),gen_text[2][19][0][-1].detach().cpu().numpy(),gen_text[2][20][0][-1].detach().cpu().numpy()])
    # neg_actis.append([gen_text[2][15][0][-1].detach().cpu().numpy(),gen_text[2][16][0][-1].detach().cpu().numpy(),gen_text[2][17][0][-1].detach().cpu().numpy()])
#################################e#######################################

positive_mean = []
negative_mean = []
sv_to_target_negative =[]
sv_to_target_positive = []
for n, layer in enumerate(INSERTION_LAYERS):
    positive_mean.append(torch.mean(torch.cat([torch.Tensor(x[0][n]) for x in positive]),0))
    negative_mean.append(torch.mean(torch.cat([torch.Tensor(x[0][n]) for x in negative]),0))
    sv_to_target_negative.append(torch.mean(torch.cat([torch.Tensor(x[0][n]) for x in negative]),0) - torch.mean(torch.cat([torch.Tensor(x[0][n]) for x in positive]),0))
    sv_to_target_positive.append(torch.mean(torch.cat([torch.Tensor(x[0][n]) for x in positive]),0) - torch.mean(torch.cat([torch.Tensor(x[0][n]) for x in negative]),0))

##################################b####################################

if "acti" in SETTING:
    positive_mean = []
    negative_mean = []

    pos_layer_0 = [a[0] for a in pos_actis]
    pos_layer_1 = [a[1] for a in pos_actis]
    pos_layer_2 = [a[2] for a in pos_actis]
    neg_layer_0 = [a[0] for a in neg_actis]
    neg_layer_1 = [a[1] for a in neg_actis]
    neg_layer_2 = [a[2] for a in neg_actis]

    positive_mean = [np.mean(pos_layer_0,0),np.mean(pos_layer_1,0),np.mean(pos_layer_2,0)]
    negative_mean = [np.mean(neg_layer_0,0),np.mean(neg_layer_1,0),np.mean(neg_layer_2,0)]
    sv_to_target_positive = [positive_mean[0] - negative_mean[0], positive_mean[1] - negative_mean[1], positive_mean[2] - negative_mean[2]]
    sv_to_target_negative = [negative_mean[0] - positive_mean[0], negative_mean[1] - positive_mean[1], negative_mean[2] - positive_mean[2]]

##################################b####################################

# with open('pos_acti.pkl', 'wb') as f:
#     pickle.dump(pos_actis,f)

# with open('neg_acti.pkl', 'wb') as f:
#     pickle.dump(neg_actis,f)

factual_prompts, subjective_prompts = load_all_sentences()

sentences_new = factual_prompts + subjective_prompts 
sentences_positive = []
sentences_negative = []
sentences_neutral = []

for sent in sentences_new:
    sentences_positive.append(sent + " Write the answer in a positive manner.")
    sentences_negative.append(sent + " Write the answer in a negative manner.")
    sentences_neutral.append(sent + " Write the answer in a neutral manner.")

# what kind of steering you want
# STEERING_SETTING = "mean"
STEERING_SETTING = "contrastive"

if STEERING_SETTING == "mean":
    selected_steer_meth_to_neg = negative_mean
    selected_steer_method_to_pos = positive_mean
else:
    selected_steer_meth_to_neg = sv_to_target_negative
    selected_steer_method_to_pos = sv_to_target_positive

# Run the steering for all type of manners
run_llm_steering(sentences_new, selected_steer_meth_to_neg, selected_steer_method_to_pos, manner="original", dataset=DATASET, setting=STEERING_SETTING, method=SETTING, tokenizer=alpaca_tokenizer, model=alpaca_model, insertion_layers=INSERTION_LAYERS, device=DEVICE, saving_path=SAVE_PATH)
# run_llm_steering(sentences_positive,selected_steer_meth_to_neg, selected_steer_method_to_pos, manner="positive", dataset=DATASET, setting=STEERING_SETTING, method=SETTING, tokenizer=alpaca_tokenizer, model=alpaca_model, insertion_layers=INSERTION_LAYERS, device=DEVICE, saving_path=SAVE_PATH)
# run_llm_steering(sentences_negative, selected_steer_meth_to_neg, selected_steer_method_to_pos, manner="negative", dataset=DATASET, setting=STEERING_SETTING, method=SETTING, tokenizer=alpaca_tokenizer, model=alpaca_model, insertion_layers=INSERTION_LAYERS, device=DEVICE, saving_path=SAVE_PATH)
# run_llm_steering(sentences_neutral, selected_steer_meth_to_neg, selected_steer_method_to_pos, manner="neutral", dataset=DATASET, setting=STEERING_SETTING, method=SETTING, tokenizer=alpaca_tokenizer, model=alpaca_model, insertion_layers=INSERTION_LAYERS, device=DEVICE, saving_path=SAVE_PATH)