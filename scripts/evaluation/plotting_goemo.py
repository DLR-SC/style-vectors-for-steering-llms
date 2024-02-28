import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pathlib import Path

from utils.load_sentences import load_all_sentences

# load environment variables
load_dotenv()

SAVE_PATH = os.getcwd()

DATASET="GoEmo"

LEGEND_FONT_SIZE = 16
FONT_SIZE=16

def mean_plots(csv_files, basic_emotions, emotion_dfs, subjective_prompts, factual_prompts, basic_emotions_w_neutral, setting, manner):
    for idx, csvfile in enumerate(csv_files):
        df = pd.read_csv(csvfile, delimiter=';')
        for jdx, emotion in enumerate(basic_emotions):
            emotion_dfs[jdx] = pd.concat([emotion_dfs[jdx], df[df['emotion'] == emotion]], ignore_index=True)

    descriptions_of_prompts=["factual", "subjective"]
    dfs_emotional_prompts = [dfe[dfe['prompt'].isin(subjective_prompts)] for dfe in emotion_dfs]
    dfs_factual_prompts = [dfe[dfe['prompt'].isin(factual_prompts)] for dfe in emotion_dfs]
    
    fig_path = os.path.join(SAVE_PATH,f"plots/eval/{DATASET}/{setting}/{manner}/")
    Path(fig_path).mkdir(parents=True, exist_ok=True)  

    for description in descriptions_of_prompts:
        if description == "factual":
            df_prompts = dfs_factual_prompts
        else:
            df_prompts = dfs_emotional_prompts
            
        for idx, emo_df in enumerate(df_prompts):
            emotion = basic_emotions[idx]

            df_ovr = emo_df[emo_df['steering_method'] != 'contrastive-neutral'].reset_index(drop=True)
            # df_neutral = emo_df[emo_df['steering_method'] == 'contrastive-neutral'].reset_index(drop=True)

            df_ovr_melt = pd.melt(df_ovr, id_vars=['lambda'], value_vars=basic_emotions_w_neutral)
            # df_neutral_melt = pd.melt(df_neutral, id_vars=['lambda'], value_vars=basic_emotions_w_neutral)

            fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))

            sns.lineplot(data=df_ovr_melt, x='lambda', y='value', hue='variable', ax=ax1)
            ax1.set_xlim(0,2.0)
            ax1.set_ylim(0,1.0)
            # ax1.set_title(f'GoEmo - factual prompts - steering to {emotion}')
            ax1.set_ylabel("Emotion class score", fontsize=FONT_SIZE)
            ax1.set_xlabel("λ", fontsize=FONT_SIZE)
            ax1.get_legend()#.remove()
            ax1.legend(fontsize=LEGEND_FONT_SIZE)
            ax1.grid()
            fig.tight_layout()
            fig.savefig(fig_path+f"{DATASET}_contrastive_steering_{setting}_{emotion}_{description}.pdf")
            plt.clf()


# def individual_plots():
#     for idx, csvfile in enumerate(csv_files):
#         df = pd.read_csv(csvfile, delimiter=';')
#         for emotion in basic_emotions:
#             df_emotion = df[df['emotion'] == emotion]
#             df_neutral = df_emotion[df_emotion['steering_method'] == 'contrastive-neutral'].reset_index(drop=True)
#             df_ovr = df_emotion[df_emotion['steering_method'] != 'contrastive-neutral'].reset_index(drop=True)
            
#             fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
#             fig.suptitle(f'Steering \"{df_neutral["prompt"][0]}\"\n towards {emotion}')

#             for emo in basic_emotions_w_neutral:
#                 ax1.plot(df_neutral['lambda'], df_neutral[emo], label=emo)
#                 ax2.plot(df_ovr['lambda'], df_ovr[emo], label=emo)

#             ax1.set_title("Contrastive-Neutral")
#             ax2.set_title("Contrastive-OVR")
#             ax1.set_xlabel(r"\lambda")
#             ax1.set_ylabel('Emotion Classifier Score')
#             ax2.set_xlabel('Lambda')
#             ax2.set_ylabel('Emotion Classifier Score')
#             ax1.legend()
#             ax2.legend()
#             plt.savefig(os.path.join(PATH_TO_REPO,f'plots/eval/{DATASET}/Go_Emo_{emotion}_{idx}.png'))


if __name__ == "__main__":
    
    setting_options = ["training_based", "activation_based_fair" , "activation_based_all"]  
    # SETTING = "training_based" # training-based style vectors
    # SETTING = "activation_based_fair" # "fair" activation-based style vectors
    SETTINGS = ["activation_based_all"] # all activation-based style vectors
    assert SETTINGS in setting_options, "Please choose the correct SETTING"

    factual_prompts, subjective_prompts = load_all_sentences()
    sents = factual_prompts + subjective_prompts

    # manner of the answers
    manners =  ["original","sad", "joyful", "fearful", "angry", "surprised", "disgusted"]
    
    for setting in SETTINGS:
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
                        
            csv_files = glob.glob(os.path.join(SAVE_PATH,f'scripts/evaluation/results/{DATASET}/{setting}/{manner}/*.csv'))
            
            # do the plots only for manners that were already computed
            if len(csv_files) == 0:
                continue

            basic_emotions = ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust']
            basic_emotions_w_neutral = ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust']

            emotion_dfs = [pd.DataFrame()] * len(basic_emotions)

            # individual_plots()
            mean_plots(csv_files, basic_emotions, emotion_dfs, sentences_subjective_manner, sentences_factual_manner, basic_emotions_w_neutral, setting, manner)
