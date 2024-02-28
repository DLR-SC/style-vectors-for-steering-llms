import pandas as pd
import glob, os

from utils.evaluation_helper import adapt_df
from utils.load_sentences import load_all_sentences
from pathlib import Path

SAVE_PATH = os.getcwd()
OUTPUT_PATH = SAVE_PATH

DATASET="shakes"

technique = "contrastive"
method = "activation_based_all"
manners = ["original"] # "positive", "negative", , "neutral"

factual_prompts, subjective_prompts = load_all_sentences()

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
    csv_files = glob.glob(os.path.join(SAVE_PATH,f"scripts/evaluation/results/{DATASET}/{method}/{technique}/{manner}/*.csv"))

    basic_emotions = ["modern", "shakes"]
    basic_emotions_w_neutral = ["modern", "shakes"]

    emotion_dfs = [pd.DataFrame()] * len(basic_emotions)
    
    if len(csv_files)>0:
        path_to_df = os.path.join(OUTPUT_PATH,f"revisit_area/{DATASET}/{method}/{type}/{manner}/")
        Path(path_to_df).mkdir(parents=True, exist_ok=True) 
    
    for file in csv_files:
        df = pd.read_csv(file)
        df = adapt_df(df, file, SAVE_PATH, method, technique, manner, dataset=DATASET) 
        direc = ""
        to = ""
        if "ToNegative" in file:
            direc = "modern"
            to = "ToNegative"
        else:
            direc = "shakes"
            to = "ToPositive"
        result = file.split(f"eval_{to}_", 1)[1]
        result = result[:-4]
        df_res = pd.DataFrame()
        df_res["input"] = len(df) * [result]
        df_res["lambda"] = list(df["lambda"])
        df_res["shakes"] = list(df["shakes"])
        df_res["modern"] = list(df["modern"])
        df_res["gen_text"] = list(df["gen_text"])
        df_res["direction"] = len(df) * [direc]
        df_res["proper_text_rating1"] = len(df) * ["yes"]
        df_res["proper_text_rating2"] = len(df) * ["yes"]
        type = ""
        if df_res.iloc[0]["input"] in sentences_factual_manner:
            df_res["type"] = len(df) * ["factual"]
            type = "factual"
        else:
            df_res["type"] = len(df) * ["subjective"]
            type = "subjective"
        # df_fact = [df[df['input_text'].isin(sentences_factual_manner)]]
        # df_subj = [df[df['input_text'].isin(sentences_subjective_manner)]]
        
        df_res.to_excel(path_to_df+f"{to}_{result[:-1].replace('?','')}.xlsx", index = None, header=True)

    
