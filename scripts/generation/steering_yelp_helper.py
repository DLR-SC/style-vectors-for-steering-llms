# pylint: disable=no-member
import os
import torch
import numpy as np
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from pathlib import Path

# Download the lexicon
nltk.download("vader_lexicon")

# Import the lexicon 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create an instance of SentimentIntensityAnalyzer
sent_analyzer = SentimentIntensityAnalyzer()


def get_sentiment(sentence):
    res = sent_analyzer.polarity_scores(sentence)
    return res["pos"], res["neg"], res["compound"], res["neu"]


def run_llm_steering(all_sentences, selected_steering_method_to_negative, selected_steering_method_to_positive, tokenizer, model, insertion_layers, device, saving_path, dataset, manner="neutral", setting="mean", method="activation_based_all"):
    """Steering of the LLMs output with the style vectors.

    :param list all_sentences: All the prompts that should be evaluated.
    :param str manner: Manner of the answer, defaults to "neutral"
    :param str setting: Setting of the steering, defaults to "mean"
    :param str method: Use activation-based or training-based style vectors, options are "activation_based_fair", "training_based", "activation_based_all", defaults to "activation_based_all"
    """
    
    for gen_run,_ in enumerate(all_sentences):
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            f"### Instruction:\r\n{all_sentences[gen_run]}\r\n\r\n### Response:"
        )
        print(f"Input:\n{all_sentences[gen_run]}")
        input_tokens = tokenizer(input_text, return_tensors="pt").to(device)

        #############
        ## sv_to_target_negative
        gen_texts = []
        prompts = []
        lmbdas = []

        pos_generated_sv_to_target_negative = []
        neg_generated_sv_to_target_negative = []
        compound_generated_sv_to_target_negative = []
        neutral_generated_sv_to_target_negative = []

        for lmd in np.linspace(0, 2, 21):
            for n, _ in enumerate(insertion_layers):
                # alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.steering_vector = nn.Parameter((lmbda * sparse_negative_sv[n]).to(device))
                if method=="training_based":
                    model.model.layers[insertion_layers[n]].mlp.steering_vector = nn.Parameter((lmd * selected_steering_method_to_negative[n]).to(device))
                else:
                    model.model.layers[insertion_layers[n]].mlp.steering_vector = nn.Parameter((lmd * torch.from_numpy(selected_steering_method_to_negative[n])).to(device))
            gen_tokens = model.generate(input_tokens.input_ids, max_length=150)
            gen_text = tokenizer.batch_decode(gen_tokens)[0].replace(input_text,'')
            pos, neg, compound, neutral = get_sentiment(gen_text)
            lmbdas.append(lmd)
            gen_texts.append(gen_text)
            pos_generated_sv_to_target_negative.append(pos)
            neg_generated_sv_to_target_negative.append(neg)
            compound_generated_sv_to_target_negative.append(compound)
            neutral_generated_sv_to_target_negative.append(neutral)
            prompts.append(input_text)
            print(f"To Negative, Lamda: {lmd} pos: {pos}, neg: {neg}, compound: {compound}")
        

        df_to_negative = pd.DataFrame()
        df_to_negative["lambda"] = lmbdas
        df_to_negative["prompt"] = prompts
        df_to_negative["gen_text"] = gen_texts
        df_to_negative["pos"] = pos_generated_sv_to_target_negative
        df_to_negative["neg"] = neg_generated_sv_to_target_negative
        df_to_negative["neutral"] = neutral_generated_sv_to_target_negative
        df_to_negative["compound"] = compound_generated_sv_to_target_negative
        df_neg = df_to_negative.set_index('lambda')
        plot_res_negative = df_neg.plot.line()
        fig = plot_res_negative.get_figure()
        fig_path = os.path.join(saving_path,f"plots/eval/{dataset}/{method}/{setting}/{manner}/")
        Path(fig_path).mkdir(parents=True, exist_ok=True) 
        fig.savefig(fig_path+f"eval_ToNegative_{all_sentences[gen_run]}.png")
        df_neg_path = os.path.join(saving_path,f"scripts/evaluation/results/{dataset}/{method}/{setting}/{manner}/")
        Path(df_neg_path).mkdir(parents=True, exist_ok=True) 
        df_neg.to_csv(df_neg_path+f"eval_ToNegative_{all_sentences[gen_run]}.csv")

        #############
        ## sv_to_target_positive
        gen_texts = []
        prompts = []
        lmbdas = []

        neutral_generated_sv_to_target_positive = []
        pos_generated_sv_to_target_positive = []
        neg_generated_sv_to_target_positive = []
        compound_generated_sv_to_target_positive = []


        for lmd in np.linspace(0, 2, 21):
            for n, _ in enumerate(insertion_layers):
                # alpaca_model.model.layers[INSERTION_LAYERS[n]].mlp.steering_vector = nn.Parameter((lmbda * sparse_negative_sv[n]).to(device))
                if method=="training_based":
                    model.model.layers[insertion_layers[n]].mlp.steering_vector = nn.Parameter((lmd * selected_steering_method_to_positive[n]).to(device))
                else:
                    model.model.layers[insertion_layers[n]].mlp.steering_vector = nn.Parameter((lmd * torch.from_numpy(selected_steering_method_to_positive[n])).to(device))
            gen_tokens = model.generate(input_tokens.input_ids, max_length=150)
            gen_text = tokenizer.batch_decode(gen_tokens)[0].replace(input_text,'')
            pos, neg, compound, neutral = get_sentiment(gen_text)
            lmbdas.append(lmd)
            gen_texts.append(gen_text)
            pos_generated_sv_to_target_positive.append(pos)
            neg_generated_sv_to_target_positive.append(neg)
            compound_generated_sv_to_target_positive.append(compound)
            neutral_generated_sv_to_target_positive.append(neutral)
            prompts.append(input_text)
            print(f"To positive, Lamda: {lmd} pos: {pos}, neg: {neg}, compound: {compound}")

        df_to_positive = pd.DataFrame()
        df_to_positive["lambda"] = lmbdas
        df_to_positive["prompt"] = prompts
        df_to_positive["gen_text"] = gen_texts
        df_to_positive["pos"] = pos_generated_sv_to_target_positive
        df_to_positive["neg"] = neg_generated_sv_to_target_positive
        df_to_positive["neutral"] = neutral_generated_sv_to_target_positive
        df_to_positive["compound"] = compound_generated_sv_to_target_positive
        df_pos = df_to_positive.set_index('lambda')
        plot_res_positive = df_pos.plot.line()
        
        fig = plot_res_positive.get_figure()
        fig_path = os.path.join(saving_path,f"plots/eval/{dataset}/{method}/{setting}/{manner}/")
        Path(fig_path).mkdir(parents=True, exist_ok=True) 
        fig.savefig(fig_path+f"eval_ToPositive_{all_sentences[gen_run]}.png")
        df_pos_path = os.path.join(saving_path,f"scripts/evaluation/results/{dataset}/{method}/{setting}/{manner}/")
        Path(df_pos_path).mkdir(parents=True, exist_ok=True) 
        df_pos.to_csv(df_pos_path+f"eval_ToPositive_{all_sentences[gen_run]}.csv")

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))
        df_neg.plot(ax=axes[0])
        df_pos.plot(ax=axes[1])

        # axes[0].text(0.5,-0.1, "\n".join(list(df_to_negative["gen_text"])), size=10, ha="center", 
        #     transform=axes[0].transAxes)
        # axes[0].text(0.5,-0.1, "\n".join(list(df_to_positive["gen_text"])), size=10, ha="center", 
        #     transform=axes[1].transAxes)
        
        plt.tight_layout()
        fig_path=os.path.join(saving_path,f"plots/eval/{dataset}/{method}/{setting}/all_directions/")
        Path(fig_path).mkdir(parents=True, exist_ok=True) 
        fig.savefig(fig_path+f"eval_bothDirections_{all_sentences[gen_run]}.png")
