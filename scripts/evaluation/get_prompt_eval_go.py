import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

from utils.load_sentences import load_all_sentences

factual_prompts, subjective_prompts = load_all_sentences()

SAVE_PATH = os.getcwd()

DATASET="GoEmo"

LEGEND_FONT_SIZE = 10
FONT_SIZE=16


def get_prompt_values_from_df(df, target_lambda):
    emotions = ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust']
    manners =  ["sad", "joyful", "fearful", "angry", "surprised", "disgusted"]
    values = {}

    for emotion, ma in zip(emotions, manners):
        try:
            # values[emotion] = df[(df['lambda'] == target_lambda) & (df['variable'] == emotion)]['value'].iloc[0]
            values[emotion] = np.mean(list(df[(df['lambda'] == target_lambda) & (df['variable'] == emotion)]['value']))
        except IndexError:
            values[emotion] = None

    return values


def get_prompt_val(subjective_prompts_p, factual_prompts_p):
    
    methods = ["activation_based_all"] #"activation_based_fair", "training_based", 

    emo_manner_p =  ["sad", "joyful", "fearful", "angry", "surprised", "disgusted"]
    emos_p = ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust']

    final_vals = {"sad": {}, "joyful":{}, "fearful": {}, "angry":{}, "surprised": {}, "disgusted":{}}
    for meth_p in methods:
        for manner_p in emo_manner_p:
            sentences_factual_manner_p = []
            sentences_subjective_manner_p = []
            if manner_p == "original":
                sentences_factual_manner_p = factual_prompts_p
                sentences_subjective_manner_p = subjective_prompts_p
            else:
                for sent in factual_prompts_p:
                    if manner_p == "angry":
                        sentences_factual_manner_p.append(sent + f" Write the answer in an {manner_p} manner.")
                    else:
                        sentences_factual_manner_p.append(sent + f" Write the answer in a {manner_p} manner.")
                    

                for sent in subjective_prompts_p:
                    if manner_p == "angry":
                        sentences_subjective_manner_p.append(sent + f" Write the answer in an {manner_p} manner.")
                    else:
                        sentences_subjective_manner_p.append(sent + f" Write the answer in a {manner_p} manner.")
                

            csv_files_p = glob.glob(os.path.join(SAVE_PATH,f"scripts/evaluation/results/{DATASET}/{meth_p}/{manner_p}/*.csv"))
            
            if len(csv_files_p) < 1:
                continue
            
            # Remove dots and question marks from the strings in the list
            cleaned_sentences = [sentence.replace('.', '').replace('?', '') for sentence in subjective_prompts_p]

            # Iterate through the CSV files and check if the cleaned sentences are present in the file names
            selected_files = []
            for csv_file in csv_files_p:
                if any(sentence in csv_file for sentence in cleaned_sentences):
                    selected_files.append(csv_file)

            csv_files_p = selected_files
            
            if len(csv_files_p) < 1:
                continue
    
            basic_emotions_p = emos_p
            basic_emotions_w_neutral_p = emos_p

            emotion_dfs_p = [pd.DataFrame()] * len(basic_emotions_p)
            for idx, csvfile in enumerate(csv_files_p):
                df = pd.read_csv(csvfile, delimiter=';')
                df["direction"] = list(df["emotion"])
                for jdx, emotion in enumerate(basic_emotions_p):
                    emotion_dfs_p[jdx] = pd.concat([emotion_dfs_p[jdx], df[df['direction'] == emotion]], ignore_index=True)
                        
            dfs_emotional_prompts = [dfe[dfe['prompt'].isin(sentences_subjective_manner_p)] for dfe in emotion_dfs_p]

            for idx, emo_df in enumerate(dfs_emotional_prompts):
                emotion = emo_df.iloc[0]["direction"]

                df_ovr = emo_df

                df_ovr_melt = pd.melt(df_ovr, id_vars=['lambda'], value_vars=basic_emotions_w_neutral_p)
            
            values = get_prompt_values_from_df(df_ovr_melt, 0)
            for em in basic_emotions_p:
                final_vals[manner_p][em] = values[em]
            
    return final_vals


def mean_plots(csv_files, basic_emotions, emotion_dfs, sentences_subjective_manner, sentences_factual_manner, basic_emotions_w_neutral, setting, manner, subjective_prompts, factual_prompts):
    for idx, csvfile in enumerate(csv_files):
        df = pd.read_csv(csvfile, delimiter=';')
        df["direction"] = list(df["emotion"])
        for jdx, emotion in enumerate(basic_emotions):
            emotion_dfs[jdx] = pd.concat([emotion_dfs[jdx], df[df['direction'] == emotion]], ignore_index=True)

    dfs_emotional_prompts = [dfe[dfe['prompt'].isin(sentences_subjective_manner)] for dfe in emotion_dfs]

    final_vals = get_prompt_val(subjective_prompts, factual_prompts)
    
    fig_path = os.path.join(SAVE_PATH,f"plots/eval/{DATASET}/{meth}/{manner}/")
    Path(fig_path).mkdir(parents=True, exist_ok=True)     
    
    for idx, emo_df in enumerate(dfs_emotional_prompts):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        emotion = emo_df.iloc[0]["direction"]

        df_ovr = emo_df

        df_ovr_melt = pd.melt(df_ovr, id_vars=['lambda'], value_vars=basic_emotions_w_neutral)


        sns.lineplot(data=df_ovr_melt, x='lambda', y='value', hue='variable', ax=axs)
        legend_handles, legend_labels_sns = axs.get_legend_handles_labels()

        sns_colors = [line.get_color() for line in legend_handles]

        axs.set_xlim(0,2)
        axs.set_ylim(0,1.0)
        
        axs.set_ylabel("Sentiment score", fontsize=FONT_SIZE)
        axs.set_xlabel("λ", fontsize=FONT_SIZE)
        emos = ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust']
        emo_manner =  ["sad", "joyful", "fearful", "angry", "surprised", "disgusted"]
        for em, ma, color in zip(emos, emo_manner, sns_colors):    
            if emotion == em:
                positivity_line = axs.axhline(y=final_vals[ma][em], color=color, linestyle='--', label=f'{emotion} (prompting)')
                break
        
        line_label = f'{emotion} (prompting)'
        legend_lines = [positivity_line]
        legend_labels = [line_label]  # Set the label here
        
        # Include the captured sns legend handles and labels into legend_lines and legend_labels
        legend_lines.extend(legend_handles)
        legend_labels.extend(legend_labels_sns)

        # Create a custom legend with the lines and the sns legend items
        legend = axs.legend(handles=legend_lines, labels=legend_labels, fontsize=LEGEND_FONT_SIZE)
        axs.add_artist(legend)  # Add the legend including the custom lines
        axs.grid()

        fig.tight_layout()
        fig.savefig(fig_path+f"{DATASET}_contrastive_subjective_source_{manner}_{emotion}_{meth}_lda1_prompt.pdf")
        fig.savefig(fig_path+f"{DATASET}_contrastive_subjective_source_{manner}_{emotion}_{meth}_lda1_prompt.png")
        plt.clf()


if __name__ == "__main__":
    technique = "contrastive"

    # "activation_based_fair" - "fair" activation-based style vectors
    # "training_based" - training-based style vectors
    # "activation_based_all" - all activation-based style vectors
    SETTINGS = ["activation_based_fair", "training_based", "activation_based_all"]

    emo_manner =  ["sad", "joyful", "fearful", "angry", "surprised", "disgusted"]
    emos = ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust']

    # manner of the answer
    manners = ["original"]+emo_manner 

    for meth in SETTINGS:
        for manner in manners:
            sentences_factual_manner = []
            sentences_subjective_manner = []
            if manner == "original":
                sentences_factual_manner = factual_prompts
                sentences_subjective_manner = subjective_prompts
            else:
                for sent in factual_prompts:
                    if manner == "angry":
                        sentences_factual_manner.append(sent + f" Write the answer in an {manner} manner.")
                    else:
                        sentences_factual_manner.append(sent + f" Write the answer in a {manner} manner.")
                    

                for sent in subjective_prompts:
                    if manner == "angry":
                        sentences_subjective_manner.append(sent + f" Write the answer in an {manner} manner.")
                    else:
                        sentences_subjective_manner.append(sent + f" Write the answer in a {manner} manner.")
            

            csv_files = glob.glob(os.path.join(SAVE_PATH,f"scripts/evaluation/results/{DATASET}/{meth}/{manner}/*.csv"))
            
            # do the plots only for manners that were already computed
            if len(csv_files) == 0:
                continue
            
            basic_emotions = ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust']
            basic_emotions_w_neutral = ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust']

            emotion_dfs = [pd.DataFrame()] * len(basic_emotions)
            
            mean_plots(csv_files, basic_emotions, emotion_dfs, sentences_subjective_manner, sentences_factual_manner, basic_emotions_w_neutral, meth, manner, subjective_prompts, factual_prompts)
