import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dotenv import load_dotenv
from pathlib import Path

from utils.load_sentences import load_all_sentences
from utils.evaluation_helper import adapt_df_plot as adapt_df

# load environment variables
load_dotenv()

SAVE_PATH = os.getcwd()
DATASET="shakes"

def mean_plots(sentences_factual_manner, sentences_subjective_manner, csv_files, emotion_dfs, subjective_prompts, factual_prompts, basic_emotions, manner, setting, technique):
    for idx, csvfile in enumerate(csv_files):
        df = pd.read_csv(csvfile, delimiter=',')
        df = adapt_df(df, csvfile, dataset=DATASET)
        for jdx, emotion in enumerate(basic_emotions):
            emotion_dfs[jdx] = pd.concat([emotion_dfs[jdx], df[df['direction'] == emotion]], ignore_index=True)

    descriptions_of_prompts=["factual", "subjective"]
    dfs_emotional_prompts = [dfe[dfe['input_text'].isin(subjective_prompts)] for dfe in emotion_dfs]
    dfs_factual_prompts = [dfe[dfe['input_text'].isin(factual_prompts)] for dfe in emotion_dfs]

    fig_path = os.path.join(SAVE_PATH,f"plots/eval/{DATASET}/{setting}/{manner}/")
    Path(fig_path).mkdir(parents=True, exist_ok=True) 
    
    for description in descriptions_of_prompts:
        if description == "factual":
            df_prompts = dfs_factual_prompts
        else:
            df_prompts = dfs_emotional_prompts
            
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for idx, emo_df in enumerate(df_prompts):
            emotion = emo_df.iloc[0]["direction"]

            df_ovr = emo_df

            df_ovr_melt = pd.melt(df_ovr, id_vars=['lambda'], value_vars=["modern", "shakes"])
            # df_ovr_melt['sample'] = range(len(df_ovr_melt))

            # Finding the samples with the highest values
            # max_value = df_ovr_melt['value'].max()
            # max_samples = df_ovr_melt[df_ovr_melt['value'] == max_value]
            
            sns.lineplot(data=df_ovr_melt, x='lambda', y='value',  hue='variable', ax=axs[idx]) #  x
            axs[idx].set_xlim(0,4.0)
            axs[idx].set_ylim(0,1.0)
            axs[idx].set_ylabel("Shakespearian style score")
            axs[idx].set_xlabel("λ")
            axs[idx].set_title(f"towards {emotion}")

            one = Line2D([0], [0], label='modern')
            two = Line2D([0], [0], label='shakespear', color='orange')

            legend = axs[idx].legend(handles=[one, two])
            # axs[idx].get_legend().remove()
            axs[idx].grid()
        fig.tight_layout()
        fig.savefig(fig_path+f"{DATASET}_contrastive_source_{setting}_{manner}_{description}.pdf")
        plt.clf()


# def individual_plots(t="subjective"):
    
#     for emotion in basic_emotions:
#         fig, ax1 = plt.subplots(1, 1, constrained_layout=True)
#         for idx, csvfile in enumerate(csv_files):
#             df = pd.read_csv(csvfile, delimiter=',')
#             df = adapt_df(df, csvfile, dataset=DATASET)
#             if t == "subjective":
#                 df = df[df['input_text'].isin(sentences_subjective_manner)] 
#             else:
#                 df = df[df['input_text'].isin(sentences_factual_manner)] 
        
#             df_emotion = df[df['direction'] == emotion]
#             df_ovr = df_emotion
            
            
#             # fig.suptitle(f'Steering \"{df_ovr["prompt"][0]}\"\n towards {emotion}')
            
#             for emo in basic_emotions_w_neutral:
#                 if emo == "pos": 
#                     col="green"
#                 else:
#                     col="red"
#                 ax1.plot(df_ovr['lambda'], df_ovr[emo], label=emo, color=col)

#             ax1.set_title(f"Yelp - steering to {emotion}")

#             ax1.set_xlabel('Lambda')
#             ax1.set_ylabel('Emotion Classifier Score')
        

#         red_patch = mpatches.Patch(color='red', label='negative sentiment')
#         blue_patch = mpatches.Patch(color='green', label='positive sentiment')

#         plt.legend(handles=[red_patch, blue_patch])
#         plt.savefig(os.path.join(PATH_TO_REPO,f'plots/eval/{DATASET}/shakes_individual_to_{emotion}_{t}.png'))


if __name__ == "__main__":
    factual_prompts, subjective_prompts = load_all_sentences()

    technique = "contrastive"
        
    setting_options = ["training_based", "activation_based_fair" , "activation_based_all"]  
    # SETTING = "training_based" # training-based style vectors
    # SETTING = "activation_based_fair" # "fair" activation-based style vectors
    SETTINGS = ["activation_based_all"]
    for setting in SETTINGS:
        assert setting in setting_options, "Please choose the correct SETTINGS"

    manners = ["positive", "negative", "original", "neutral"]

    for setting in SETTINGS:
        for manner in manners:
            # sentences_factual_manner = []
            # sentences_subjective_manner = []
            # if manner == "original":
            #     sentences_factual_manner = factual_prompts
            #     sentences_subjective_manner = subjective_prompts
            # else:
            #     for sent in factual_prompts:
            #         sentences_factual_manner.append(sent + f" Write the answer in a {manner} manner.")

            #     for sent in subjective_prompts:
            #         sentences_subjective_manner.append(sent + f" Write the answer in a {manner} manner.")
            
            csv_files = glob.glob(os.path.join(SAVE_PATH,f"scripts/evaluation/results/{DATASET}/{setting}/{technique}/{manner}/*.csv"))

            if len(csv_files)<1:
                continue
            
            basic_emotions = ["modern", "shakes"]
            basic_emotions_w_neutral = ["modern", "shakes"]

            emotion_dfs = [pd.DataFrame()] * len(basic_emotions)
            
            # individual_plots(t="factual")
            mean_plots(factual_prompts, subjective_prompts, csv_files, emotion_dfs, subjective_prompts, factual_prompts, basic_emotions, manner, setting, technique)
