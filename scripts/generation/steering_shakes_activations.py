# pylint: disable=no-member
import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

from utils.load_sentences import load_all_sentences
from utils.steering_vector_loader import load_activations_shake
from utils.llm_model_utils import load_llm_model_with_insertions

# load environment variables
load_dotenv()

shakes_classifier = pipeline("text-classification", model="notaphoenix/shakespeare_classifier_model", top_k=None)

DATASET = "shakes"
SETTING = "activation_based_all"

SAVE_PATH=os.getcwd()

INSERTION_LAYERS = [18,19,20]
DEVICE = torch.device('cuda:1')
alpaca_model, alpaca_tokenizer = load_llm_model_with_insertions(DEVICE, INSERTION_LAYERS)


# load the activation vectors
VECTOR_PATH = os.getenv("ACTIVATIONS_PATH_Shake")
steering_vectors = load_activations_shake(VECTOR_PATH, DATASET)

# positive is modern, negative is original
positive = [sv for sv in steering_vectors if sv[-1] == 1]
negative = [sv for sv in steering_vectors if sv[-1] == 0]

################################b########################################
pos_actis = []
neg_actis = []
for sv in tqdm(positive):
    input_tokens = alpaca_tokenizer(sv[1].replace('\n',''), return_tensors="pt").to(DEVICE)
    gen_text = alpaca_model.forward(input_tokens.input_ids, output_hidden_states=True)
    # pos_actis.append(gen_text[2][15].detach().cpu().numpy()[0])
    pos_actis.append([gen_text[2][18][0][-1].detach().cpu().numpy(),gen_text[2][19][0][-1].detach().cpu().numpy(),gen_text[2][20][0][-1].detach().cpu().numpy()])
    # pos_actis.append([gen_text[2][15][0][-1].detach().cpu().numpy(),gen_text[2][16][0][-1].detach().cpu().numpy(),gen_text[2][17][0][-1].detach().cpu().numpy()])

for sv in tqdm(negative):
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
    positive_mean.append(torch.mean(torch.cat([torch.from_numpy(x[0][n]) for x in positive]),0))
    negative_mean.append(torch.mean(torch.cat([torch.from_numpy(x[0][n]) for x in negative]),0))
    sv_to_target_negative.append(torch.mean(torch.cat([torch.from_numpy(x[0][n]) for x in negative]),0) - torch.mean(torch.cat([torch.from_numpy(x[0][n]) for x in positive]),0))
    sv_to_target_positive.append(torch.mean(torch.cat([torch.from_numpy(x[0][n]) for x in positive]),0) - torch.mean(torch.cat([torch.from_numpy(x[0][n]) for x in negative]),0))

##################################b####################################

positive_mean = []
negative_mean = []

pos_layer_15 = [a[0] for a in pos_actis]
pos_layer_16 = [a[1] for a in pos_actis]
pos_layer_17 = [a[2] for a in pos_actis]
neg_layer_15 = [a[0] for a in neg_actis]
neg_layer_16 = [a[1] for a in neg_actis]
neg_layer_17 = [a[2] for a in neg_actis]

positive_mean = [np.mean(pos_layer_15,0),np.mean(pos_layer_16,0),np.mean(pos_layer_17,0)]
negative_mean = [np.mean(neg_layer_15,0),np.mean(neg_layer_16,0),np.mean(neg_layer_17,0)]
sv_to_target_positive = [positive_mean[0] - negative_mean[0], positive_mean[1] - negative_mean[1], positive_mean[2] - negative_mean[2]]
sv_to_target_negative = [negative_mean[0] - positive_mean[0], negative_mean[1] - positive_mean[1], negative_mean[2] - positive_mean[2]]


##################################b####################################
# original == 0
# modern == 1

# with open('pos_acti.pkl', 'wb') as f:
#     pickle.dump(pos_actis,f)

# with open('neg_acti.pkl', 'wb') as f:
#     pickle.dump(neg_actis,f)

factual_prompts, subjective_prompts = load_all_sentences()
sentences_new = factual_prompts + subjective_prompts


def run(all_sentences, manner="neutral", setting="mean", method="activation_based_all"):
    if setting == "mean" or setting == "new_mean":
        selected_steering_method_to_negative = negative_mean
        selected_steering_method_to_positive = positive_mean
    else:
        selected_steering_method_to_negative = sv_to_target_negative
        selected_steering_method_to_positive = sv_to_target_positive

    for gen_run,_ in enumerate(all_sentences):
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            f"### Instruction:\r\n{all_sentences[gen_run]}\r\n\r\n### Response:"
        )
        print(f"Input:\n{all_sentences[gen_run]}")
        input_tokens = alpaca_tokenizer(input_text, return_tensors="pt").to(DEVICE)

        #############
        ## sv_to_target_negative
        gen_texts = []
        prompts = []
        lmbdas = []

        pos = []
        neg = []

        for lmd in np.linspace(0, 4, 11):
            for n, _ in enumerate(INSERTION_LAYERS):
                # alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.steering_vector = nn.Parameter((lmbda * sparse_negative_sv[n]).to(device))
                if setting == "mean" or setting == "contrastive":
                    alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.shift_with_new_idea = False
                    alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.steering_vector = nn.Parameter((lmd * torch.from_numpy(selected_steering_method_to_negative[n])).to(DEVICE))
                else:#
                    alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.shift_with_new_idea = True
                    alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.b = lmd
                    alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.steering_vector = nn.Parameter(torch.from_numpy(selected_steering_method_to_negative[n]).to(DEVICE))
            gen_tokens = alpaca_model.generate(input_tokens.input_ids, max_length=150)
            gen_text = alpaca_tokenizer.batch_decode(gen_tokens)[0].replace(input_text,'')
            shakes_class = shakes_classifier(gen_text)[0]

            if shakes_class[0]["label"] == "modern":
                p = shakes_class[0]["score"]
                n = shakes_class[1]["score"]
                pos.append(p)
                neg.append(n)
            else:
                p = shakes_class[1]["score"]
                n = shakes_class[0]["score"]
                pos.append(p)
                neg.append(n)
            lmbdas.append(lmd)
            gen_texts.append(gen_text)

            prompts.append(input_text)
            print(f"To modern, Lamda: {lmd} modern: {p}, shakes: {n}")
        

        df_to_negative = pd.DataFrame()
        df_to_negative["lamda"] = lmbdas
        df_to_negative["prompt"] = prompts
        df_to_negative["gen_text"] = gen_texts
        df_to_negative["modern"] = pos
        df_to_negative["shakes"] = neg

        df_neg = df_to_negative.set_index('lamda')
        plot_res_negative = df_neg.plot.line()
        fig = plot_res_negative.get_figure()
        fig_path=os.path.join(SAVE_PATH,f"plots/eval/{DATASET}/{method}/{setting}/{manner}/")
        Path(fig_path).mkdir(parents=True, exist_ok=True) 
        fig.savefig(fig_path+f"eval_ToNegative_{all_sentences[gen_run]}.png")
        df_neg_path=os.path.join(SAVE_PATH,f"scripts/evaluation/results/{DATASET}/{method}/{setting}/{manner}/")
        Path(df_neg_path).mkdir(parents=True, exist_ok=True) 
        df_neg.to_csv(df_neg_path+f"eval_ToNegative_{all_sentences[gen_run]}.csv")

        #############
        ## sv_to_target_positive
        gen_texts = []
        prompts = []
        lmbdas = []

        pos = []
        neg = []

        for lmd in np.linspace(0, 4, 11):
            for n, _ in enumerate(INSERTION_LAYERS):
                # alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.steering_vector = nn.Parameter((lmbda * sparse_negative_sv[n]).to(device))
                if setting == "mean" or setting == "contrastive":
                    alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.shift_with_new_idea = False
                    alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.steering_vector = nn.Parameter((lmd * torch.from_numpy(selected_steering_method_to_positive[n])).to(DEVICE))
                else:#
                    alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.shift_with_new_idea = True
                    alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.b = lmd
                    alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.steering_vector = nn.Parameter(torch.from_numpy(selected_steering_method_to_positive[n]).to(DEVICE))
            gen_tokens = alpaca_model.generate(input_tokens.input_ids, max_length=150)
            gen_text = alpaca_tokenizer.batch_decode(gen_tokens)[0].replace(input_text,'')
            shakes_class = shakes_classifier(gen_text)[0]

            if shakes_class[0]["label"] == "modern":
                p = shakes_class[0]["score"]
                n = shakes_class[1]["score"]
                pos.append(p)
                neg.append(n)
            else:
                p = shakes_class[1]["score"]
                n = shakes_class[0]["score"]
                pos.append(p)
                neg.append(n)
            lmbdas.append(lmd)
            gen_texts.append(gen_text)

            prompts.append(input_text)
            print(f"To shakes, Lamda: {lmd} modern: {p}, shakes: {n}")

        df_to_positive = pd.DataFrame()
        df_to_positive["lamda"] = lmbdas
        df_to_positive["prompt"] = prompts
        df_to_positive["gen_text"] = gen_texts
        df_to_positive["modern"] = pos
        df_to_positive["shakes"] = neg

        df_pos = df_to_positive.set_index('lamda')
        plot_res_positive = df_pos.plot.line()
        
        fig = plot_res_positive.get_figure()
        fig_path = os.path.join(SAVE_PATH,f"plots/eval/{DATASET}/{method}/{setting}/{manner}/")
        Path(fig_path).mkdir(parents=True, exist_ok=True) 
        fig.savefig(fig_path+f"eval_ToPositive_{all_sentences[gen_run]}.png")
        df_pos_save = os.path.join(SAVE_PATH,f"scripts/evaluation/results/{DATASET}/{method}/{setting}/{manner}/")
        Path(df_pos_save).mkdir(parents=True, exist_ok=True) 
        df_pos.to_csv(df_pos_save+f"eval_ToPositive_{all_sentences[gen_run]}.csv")

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))
        df_neg.plot(ax=axes[0])
        df_pos.plot(ax=axes[1])

        # axes[0].text(0.5,-0.1, "\n".join(list(df_to_negative["gen_text"])), size=10, ha="center", 
        #     transform=axes[0].transAxes)
        # axes[0].text(0.5,-0.1, "\n".join(list(df_to_positive["gen_text"])), size=10, ha="center", 
        #     transform=axes[1].transAxes)
        
        plt.tight_layout()
        fig_path = os.path.join(SAVE_PATH,f"plots/eval/{DATASET}/{method}/{setting}/all_directions/")
        Path(fig_path).mkdir(parents=True, exist_ok=True) 
        fig.savefig(fig_path+f"eval_bothDirections_{all_sentences[gen_run]}.png") 


run(sentences_new, manner="original", setting="contrastive", method=SETTING)