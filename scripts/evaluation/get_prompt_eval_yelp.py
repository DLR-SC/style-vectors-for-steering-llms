import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from utils.load_sentences import load_all_sentences

SAVE_PATH = os.getcwd()

legend_font_size = 10
font_size=16

DATASET="yelp"

def adapt_df(df, csvfile):
    direc = ""
    to = ""
    if "ToNegative" in csvfile:
        direc = "neg"
        to = "ToNegative"
    else:
        direc = "pos"
        to = "ToPositive"
    s = csvfile
    
    result = s.split(f"eval_{to}_", 1)[1]

    # result = re.search(f"eval_{to}_(.*)", result)
    # result = result.group(1)
    input_text = len(df) * [result[:-4]]
    direction = len(df) * [direc]
    df["direction"] = direction
    df["input_text"] = input_text

    df = df.rename(columns={'lamda': 'lambda'})
    return df

def get_prompt_values_from_df(df, target_lambda):
    try:
        value_pos = np.mean(list(df[(df['lambda'] == target_lambda) & (df['variable'] == 'pos')]['value']))
    except IndexError:
        value_pos = None
        
    try:
        value_neg = np.mean(list(df[(df['lambda'] == target_lambda) & (df['variable'] == 'neg')]['value']))
    except IndexError:
        value_neg = None
        
    return value_pos, value_neg


def get_prompt_val(factual_prompts_p, subjective_prompts_p, technique_p):
    
    methods = ["activation_based_all"] #"activation_based_fair", "training_based", 

    manners_p = ["positive", "negative"] #   ,"neutral"
    # manners = ["original"]

    single_vals = {"positive": {}, "negative":{}}
    final_vals = {"positive": {}, "negative":{}}
    for meth in methods:
        for manner_p in manners_p:
            sentences_factual_manner_p = []
            sentences_subjective_manner_p = []
            if manner_p == "original":
                sentences_factual_manner_p = factual_prompts_p
                sentences_subjective_manner_p = subjective_prompts_p
            else:
                for sent in factual_prompts_p:
                    sentences_factual_manner_p.append(sent + f" Write the answer in a {manner_p} manner.")

                for sent in subjective_prompts_p:
                    sentences_subjective_manner_p.append(sent + f" Write the answer in a {manner_p} manner.")
            
            csv_path_p = os.path.join(SAVE_PATH,f"scripts/evaluation/results/{DATASET}/{meth}/{technique_p}/{manner_p}/")
            csv_files_p = glob.glob(csv_path_p + "*.csv")
            
            if len(csv_files_p) < 1:
                continue 
            
            basic_emotions_p = ["pos", "neg"]
            basic_emotions_w_neutral_p = ["pos", "neg"]

            emotion_dfs_p = [pd.DataFrame()] * len(basic_emotions_p)
            for idx, csvfile in enumerate(csv_files_p):
                df = pd.read_csv(csvfile, delimiter=',')
                df = adapt_df(df, csvfile)
                for jdx, emotion in enumerate(basic_emotions_p):
                    emotion_dfs_p[jdx] = pd.concat([emotion_dfs_p[jdx], df[df['direction'] == emotion]], ignore_index=True)
            
            dfs_emotional_prompts = [dfe[dfe['input_text'].isin(sentences_subjective_manner_p)] for dfe in emotion_dfs_p]

            for idx, emo_df in enumerate(dfs_emotional_prompts):
                emotion = emo_df.iloc[0]["direction"]

                df_ovr = emo_df

                df_ovr_melt = pd.melt(df_ovr, id_vars=['lambda'], value_vars=basic_emotions_w_neutral_p)
            
            pos, neg = get_prompt_values_from_df(df_ovr_melt, 0)
            final_vals[manner_p]["positivity"] = pos
            final_vals[manner_p]["negativity"] = neg
    
            single_vals_ = df_ovr_melt[df_ovr_melt["lambda"] == 0.0]

            single_vals[manner_p]["positivity"] = list(single_vals_[single_vals_["variable"] == "pos"].value)
            single_vals[manner_p]["negativity"] = list(single_vals_[single_vals_["variable"] == "neg"].value)
    return final_vals, single_vals


def mean_plots(sentences_factual_manner, sentences_subjective_manner, basic_emotions_w_neutral, csv_files, basic_emotions, emotion_dfs, manner, meth, technique, factual_prompts, subjective_prompts):
    for idx, csvfile in enumerate(csv_files):
        df = pd.read_csv(csvfile, delimiter=',')
        df = adapt_df(df, csvfile)
        for jdx, emotion in enumerate(basic_emotions):
            emotion_dfs[jdx] = pd.concat([emotion_dfs[jdx], df[df['direction'] == emotion]], ignore_index=True)

    fig_path = os.path.join(SAVE_PATH,f"plots/eval/{DATASET}/{meth}/{manner}/")
    Path(fig_path).mkdir(parents=True, exist_ok=True)     
    
    dfs_emotional_prompts = [dfe[dfe['input_text'].isin(sentences_subjective_manner)] for dfe in emotion_dfs]
    dfs_factual_prompts = [dfe[dfe['input_text'].isin(sentences_factual_manner)] for dfe in emotion_dfs]

    # fig, axs = plt.subplots(2, 1, figsize=(5, 8), dpi=120)
    # for idx, emo_df in enumerate(dfs_factual_prompts):
    #     emotion = emo_df.iloc[0]["direction"]

    #     df_ovr = emo_df

    #     df_ovr_melt = pd.melt(df_ovr, id_vars=['lambda'], value_vars=basic_emotions_w_neutral)
    #     df_ovr_melt['sample'] = range(len(df_ovr_melt))

    #     # Finding the samples with the highest values
    #     max_value = df_ovr_melt['value'].max()
    #     max_samples = df_ovr_melt[df_ovr_melt['value'] == max_value]
        
    #     sns.lineplot(data=df_ovr_melt, x='lambda', y='value',  hue='variable', ax=axs[idx]) #  x
    #     axs[idx].set_xlim(0,1.2)
    #     axs[idx].set_ylim(0,1.0)
    #     # axs[idx].set_title(f'Yelp - Steering direction: {emotion}')
    #     print_method = ""
    #     if meth == "training_based":
    #         print_method = "trained vector based"
    #     elif meth == "activation_based_all":
    #         print_method = "activation based all"
    #     else:
    #         print_method = "activation based fair"

    #     # if emotion == "pos":
    #     #     axs[idx].set_title(f"Yelp - {print_method} - steering to positive")
    #     # else:
    #     #     axs[idx].set_title(f"Yelp - {print_method} - steering to negative")
    #     axs[idx].set_ylabel("Sentiment score", fontsize=font_size)
    #     axs[idx].set_xlabel("λ", fontsize=font_size)

    #     one = Line2D([0], [0], label='positive')
    #     two = Line2D([0], [0], label='negative', color='orange')

    #     legend = axs[idx].legend(handles=[one, two], fontsize=legend_font_size)
    #     # axs[idx].get_legend()
    #     axs[idx].grid()
    # fig.tight_layout()

    # fig.savefig(os.path.join(PATH_TO_REPO,f"plots/eval/{DATASET}/yelp_contrastive_factual_source_{manner}_{meth}_lda1.pdf"))
    # plt.clf()
    final_vals, single_vals = get_prompt_val(factual_prompts, subjective_prompts, technique)
    
    df_table = pd.DataFrame()
    fig, axs = plt.subplots(2, 1, figsize=(5, 8), dpi=120)
    for idx, emo_df in enumerate(dfs_emotional_prompts):
        emotion = emo_df.iloc[0]["direction"]

        df_ovr = emo_df

        df_ovr_melt = pd.melt(df_ovr, id_vars=['lambda'], value_vars=basic_emotions_w_neutral)

        sns.lineplot(data=df_ovr_melt, x='lambda', y='value', hue='variable', ax=axs[idx])
        axs[idx].set_xlim(0,1.2)
        axs[idx].set_ylim(0,1.0)

        # df_ovr_melt.to_csv(fig_path+f"df_for_you_{meth}_{emotion}.csv")

        axs[idx].set_ylabel("Sentiment score", fontsize=font_size)
        axs[idx].set_xlabel("λ", fontsize=font_size)

        if emotion == "pos":
            positivity_line = axs[idx].axhline(y=final_vals["positive"]["positivity"], color='tab:blue', linestyle='--', label='positive (prompting)')
            negativity_line = axs[idx].axhline(y=final_vals["positive"]["negativity"], color='tab:orange', linestyle='--', label='negative (prompting)')
        elif emotion == "neg":
            positivity_line = axs[idx].axhline(y=final_vals["negative"]["positivity"], color='tab:blue', linestyle='--', label='positive (prompting)')
            negativity_line = axs[idx].axhline(y=final_vals["negative"]["negativity"], color='tab:orange', linestyle='--', label='negative (prompting)')

        one = Line2D([0], [0], color='tab:blue', linestyle='-', label='positive')
        two = Line2D([0], [0], color='tab:orange', linestyle='-', label='negative')

        # Create a custom legend with the lines and the horizontal lines
        legend_lines = [positivity_line, negativity_line, one, two]
        legend_labels = [line.get_label() for line in legend_lines]
        legend = axs[idx].legend(handles=legend_lines, labels=legend_labels, fontsize=legend_font_size)
        axs[idx].add_artist(legend)  # Add the legend including the custom lines
        axs[idx].grid()

        df_table["sentence"] = sentences_subjective_manner
        if emotion == "pos":
            df_table["prompt_positive_steering_positivity"] = single_vals["positive"]["positivity"] #[final_vals["positive"]["positivity"]] * len(sentences_subjective_manner)
            df_table["prompt_positive_steering_negativity"] = single_vals["positive"]["negativity"] #[final_vals["positive"]["negativity"]] * len(sentences_subjective_manner)

            

            for i in list(emo_df["lambda"].unique()):
                df_table[f"{meth}_positive_steering_positivity_lbd{i}"] = list(emo_df[emo_df["lambda"] == i]["pos"])
                df_table[f"{meth}_positive_steering_negativity_lbd{i}"] = list(emo_df[emo_df["lambda"] == i]["neg"])

        else:
            df_table["prompt_negative_steering_positivity"] = single_vals["negative"]["positivity"]  # [final_vals["negative"]["positivity"]] * len(sentences_subjective_manner)
            df_table["prompt_negative_steering_negativity"] = single_vals["negative"]["negativity"] # [final_vals["negative"]["negativity"]] * len(sentences_subjective_manner)

            for i in list(emo_df["lambda"].unique()):
                df_table[f"{meth}_positive_steering_positivity_lbd{i}"] = list(emo_df[emo_df["lambda"] == i]["pos"])
                df_table[f"{meth}_positive_steering_negativity_lbd{i}"] = list(emo_df[emo_df["lambda"] == i]["neg"])
        
        # df_table["sentence"] = meth
        # one = Line2D([0], [0], label='positive')
        # two = Line2D([0], [0], label='negative', color='orange')

        # legend = axs[idx].legend(handles=[one, two], fontsize=legend_font_size)
        # # axs[idx].get_legend().remove()
        # axs[idx].grid()
    
        # if emotion == "pos":
        #     axs[idx].axhline(y=final_vals["positive"]["positivity"], color='tab:blue', linestyle='--', label='Positivity of prompt baseline')
        #     axs[idx].axhline(y=final_vals["positive"]["negativity"], color='tab:orange', linestyle='--', label='Negativity of prompt baseline')
        # elif emotion == "neg":
        #     axs[idx].axhline(y=final_vals["negative"]["positivity"], color='tab:blue', linestyle='--', label='Positivity of prompt baseline')
        #     axs[idx].axhline(y=final_vals["negative"]["negativity"], color='tab:orange', linestyle='--', label='Negativity of prompt baseline')
        # axs[idx].legend(fontsize=legend_font_size)
    
    fig.tight_layout()    
    # df_table.to_csv(fig_path+f"table_for_you_{meth}.csv"))
    fig.savefig(fig_path+f"{DATASET}_contrastive_subjective_source_{manner}_{emotion}_{meth}_lda1_prompt.pdf")
    fig.savefig(fig_path+f"{DATASET}_contrastive_subjective_source_{manner}_{emotion}_{meth}_lda1_prompt.png")
    plt.clf()


if __name__ == "__main__":
    technique = "contrastive"
    
    setting_options = ["training_based", "activation_based_fair" , "activation_based_all"] 
    # "activation_based_fair" - "fair" activation-based style vectors
    # "training_based" - training-based style vectors
    # "activation_based_all" - all activation-based style vectors
    SETTINGS = ["activation_based_fair", "training_based", "activation_based_all"]
    for setting in SETTINGS:
        assert setting in setting_options, "Please choose the correct SETTINGS"

    # the manner of the answer
    manners = [ "original","positive", "negative","neutral"]

    factual_prompts, subjective_prompts = load_all_sentences()  

    for setting in SETTINGS:
        for manner in manners:
            sentences_factual_manner = []
            sentences_subjective_manner = []
            if manner == "original":
                sentences_factual_manner = factual_prompts
                sentences_subjective_manner = subjective_prompts
            else:
                for sent in factual_prompts:
                    sentences_factual_manner.append(sent + f" Write the answer in a {manner} manner.")

                for sent in subjective_prompts:
                    sentences_subjective_manner.append(sent + f" Write the answer in a {manner} manner.")
            

            path_to_csv=os.path.join(SAVE_PATH,f"scripts/evaluation/results/{DATASET}/{setting}/{technique}/{manner}/")
            csv_files = glob.glob(path_to_csv+"*.csv")
            
            if len(csv_files)==0:
                continue
            
            basic_emotions = ["pos", "neg"]
            basic_emotions_w_neutral = ["pos", "neg"]

            emotion_dfs = [pd.DataFrame()] * len(basic_emotions)
            
            # individual_plots(t="subjective")
            mean_plots(sentences_factual_manner, sentences_subjective_manner, basic_emotions_w_neutral, csv_files, basic_emotions, emotion_dfs, manner, setting, technique, factual_prompts, subjective_prompts)
