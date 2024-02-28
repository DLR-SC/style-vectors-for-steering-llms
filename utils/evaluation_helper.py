import os
import re 

def adapt_df(df, csvfile, PATH_TO_REPO, method, technique, manner, dataset):
    direc = ""
    to = ""
    if "ToNegative" in csvfile:
        if dataset=="yelp":
            direc = "neg"
        elif dataset=="shakes":
            direc = "modern"
        to = "ToNegative"
    else:
        if dataset=="yelp":
            direc = "pos"
        elif dataset=="shakes":
            direc = "shakes"
        to = "ToPositive"
    s = csvfile
    result = re.search(os.path.join(PATH_TO_REPO,f"scripts/evaluation/results/{dataset}/{method}/{technique}/{manner}/(.*).csv"), s)
    result = result.group(1)
    
    result = result.split(f"eval_{to}_", 1)[1]

    # result = re.search(f"eval_{to}_(.*)", result)
    # result = result.group(1)
    input_text = len(df) * [result]
    direction = len(df) * [direc]
    df["direction"] = direction
    df["input_text"] = input_text

    df = df.rename(columns={'lamda': 'lambda'})
    return df


def adapt_df_plot(df, csvfile, dataset):
    direc = ""
    to = ""
    if "ToNegative" in csvfile:
        if dataset=="yelp":
            direc = "neg"
        elif dataset=="shakes":
            direc = "modern"
        to = "ToNegative"
    else:
        if dataset=="yelp":
            direc = "pos"
        elif dataset=="shakes":
            direc = "shakes"
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