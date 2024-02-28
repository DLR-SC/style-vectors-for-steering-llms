# pylint: disable=no-member
import os
import torch
import numpy as np
from torch import nn
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path

def concat_layers_tv(go_emo_train_tv):
    for idx, entry in enumerate(go_emo_train_tv):
        concatenated_layers = np.concatenate(entry[2])
        go_emo_train_tv[idx][2] = concatenated_layers
    
    return go_emo_train_tv

def calculate_means_tv(go_emo_train_tv, labels):
    """return means, ovr_r_means, total_mean for trained vectors"""
    means, total_mean, ovr_r_means = [], [], []
    concat_layers_tv(go_emo_train_tv)
    
    for label in labels:
        label_samples = [entry[2] for entry in go_emo_train_tv if int(entry[1]['labels'][0]) == label]
        # label_samples += [entry[2] for entry in go_emo_test_tv if entry[1]['labels'][0] == label]
        r_labels = [entry[2] for entry in go_emo_train_tv if int(entry[1]['labels'][0]) != label]
        # r_labels += [entry[2] for entry in go_emo_test_tv if entry[1]['labels'][0] != label]
        means.append(np.mean(label_samples,0))
        ovr_r_means.append(np.mean(r_labels,0))
    total_mean.append(np.mean(means,0))
    
    return means, ovr_r_means, total_mean

def concat_layers(go_emo_train, go_emo_test, insertion_layers):
    for idx, entry in enumerate(go_emo_train):
        concatenated_layers = np.concatenate(entry[2][insertion_layers[0]:insertion_layers[-1]+1])
        go_emo_train[idx][2] = concatenated_layers
    for idx, entry in enumerate(go_emo_test):
        concatenated_layers = np.concatenate(entry[2][insertion_layers[0]:insertion_layers[-1]+1])
        go_emo_test[idx][2] = concatenated_layers        
    return go_emo_train, go_emo_test


def calculate_means(go_emo_train, go_emo_test, labels, insertion_layers, mode="all", go_emo_train_act_tv=None):
    """return means, ovr_r_means, total_mean"""    
    means, total_mean, ovr_r_means = [], [], []
    go_emo_train, go_emo_test = concat_layers(go_emo_train, go_emo_test, insertion_layers)
    
    for label in labels:
        if mode=="fair":
            label_samples = [entry[2] for entry in go_emo_train_act_tv if entry[1]['labels'][0] == label]
            r_labels = [entry[2] for entry in go_emo_train_act_tv if entry[1]['labels'][0] != label]
        else:
            label_samples = [entry[2] for entry in go_emo_train if entry[1]['labels'][0] == label]
            label_samples += [entry[2] for entry in go_emo_test if entry[1]['labels'][0] == label]
            r_labels = [entry[2] for entry in go_emo_train if entry[1]['labels'][0] != label]
            r_labels += [entry[2] for entry in go_emo_test if entry[1]['labels'][0] != label]
        
        means.append(np.mean(label_samples,0))
        ovr_r_means.append(np.mean(r_labels,0))
    total_mean.append(np.mean(means,0))
    
    return means, ovr_r_means, total_mean

def get_distilroberta_classifier():
    """Get the emotion-english-distilroberta-base classifier."""
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    return classifier

def eval_all_prompts_all_emos(evaluation_prompts, manners, emotions, means, ovr_r_means, total_mean, setting, llm_model, tokenizer, insertion_layers, save_path, device, dataset):
    """Evaluate all prompts in the specified emotional manners for all emotions for all lambda steering strengths.

    :param list evaluation_prompts: List of all prompts.
    :param list manners: The llm should answer in these manners.
    :param list emotions: List of all emotions we want to evaluate.
    :param _type_ means: _description_
    :param _type_ ovr_r_means: _description_
    :param _type_ total_mean: _description_
    :param str setting: _description_
    :param llm_model: LLM model
    :param tokenizer: Tokenizer for the LLM
    :param list insertion_layers: List of integers
    :param str save_path: Where to save.
    :param device: Cuda device.
    """

    
    # get the classifier for the emotion classification
    classifier = get_distilroberta_classifier()
    
    # iterate over all manners
    for manner in tqdm(manners, total=len(manners)):
        # iterate over all sentences
        for num_sentence, sentence in enumerate(evaluation_prompts):
            if manner == "angry":
                sentence = sentence + f" Write the answer in an {manner} manner."
            elif manner == "original":
                sentence = sentence
            else:
                sentence = sentence + f" Write the answer in a {manner} manner."
            user_input = sentence
            input_text = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\r\n\r\n"
                    f"### Instruction:\r\n{user_input}\r\n\r\n### Response:"
                )

            input_tokens = tokenizer(input_text, return_tensors="pt").to(device)
            csv_dump = [['lambda', 'emotion', 'prompt', 'gen_text','steering_method', 'sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust', 'neutral']]
            #lamda,prompt,gen_text
            # iterate over all emotions we want to evaluate
            for idx, emotion in enumerate(emotions):
                emo_mean = means[idx]
                emo_ovr_mean = ovr_r_means[idx]

                sv_to_target_emotion_ovr = np.split(emo_mean - emo_ovr_mean, len(insertion_layers))
                sv_to_target_emotion_total_mean = np.split(emo_mean - total_mean[0] ,len(insertion_layers))
                sv_target_emotion = np.split(emo_mean, len(insertion_layers))
                
                svs = [sv_to_target_emotion_ovr]
                svs_string = ["contrastive-OVR"]
                
                for k,_ in enumerate(svs):
                    # iterate over all lambas from 0 to 2
                    for i in np.linspace(0.0, 2.0, 11):
                        lmbda = i 
                        for n, _ in enumerate(insertion_layers):
                            llm_model.model.layers[insertion_layers[n]].mlp.steering_vector = nn.Parameter(torch.from_numpy(svs[k][n]).to(device))
                            llm_model.model.layers[insertion_layers[n]].mlp.b = lmbda
                            
                        gen_tokens = llm_model.generate(input_tokens.input_ids, max_length=150)
                        # print("##########################################################################################")
                        print(f"Steering sentence \"{sentence}\" towards {emotion}, coefficient {lmbda}, method {svs_string[k]}")
                        # print(f"Using {svs_string[k]} steering vector with coefficient {lmbda}")
                        output = tokenizer.batch_decode(gen_tokens)[0].replace(input_text,'').replace('\n', ' ').replace(';','-')
                        print(f"Generated sentence: {output}")
                        print("##########################################################################################")
                        sentence_classification = classifier(output)
                        csv_dump.append([str(lmbda), emotion, sentence, output, svs_string[k], 
                                        str(sentence_classification[0][5]['score']),
                                        str(sentence_classification[0][3]['score']),
                                        str(sentence_classification[0][2]['score']),
                                        str(sentence_classification[0][0]['score']),
                                        str(sentence_classification[0][6]['score']),
                                        str(sentence_classification[0][1]['score']),
                                        str(sentence_classification[0][4]['score'])])
            save_path_full = os.path.join(save_path,f"scripts/evaluation/results/{dataset}/{setting}/{manner}/")
            Path(save_path_full).mkdir(parents=True, exist_ok=True) 
            np.savetxt(save_path_full+f"Go_Emotions_{sentence.replace('?', '')}.csv", csv_dump, delimiter=";", fmt='%s')
            #HERE SAVE TO CSV