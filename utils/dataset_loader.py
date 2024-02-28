import glob, logging, sys, os
import pandas as pd 
from datasets import load_dataset
from tqdm import tqdm
from utils.data_prep import goemo_get_only_ekman

#=================================================================#
# We are using dataframes with the dataloader for the model. Therefore, we have to create these dataframes from the raw dataset files.
### ### ### PLEASE SPECIFY ### ### ###
path_datasets = "/localdata1/EmEx/datasets/" # path to folder with loaded datasets
path_dataframes = "/localdata1/EmEx/datasets/dataframes/" # path to save and load dataframes 

load_git = False # change to true, if you want missing repositories to be downloaded directly
#=================================================================#
### ### ### set up logger ### ### ###
logger = logging.getLogger('dataset_logger')

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.WARNING)
c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)

f_handler = logging.FileHandler('./dataset_loader.log')
f_handler.setLevel(logging.ERROR)
f_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(f_handler)


### ### ### Util functions ### ### ###
def sysprint(f, text):
    sys.stdout.write(F"dataset_loader INFO - {f}: \n{text}\n\n")

def check_path(path): # check if the paths exists, else create 
    if not os.path.exists(path):
        os.makedirs(path)
        print("created path "+ path)
    return path

check_path(path_dataframes) # create the dataframe folder if it does not exist yet


#=================================================================#
### Sentiment Positive↔Negative Yelp14 (Shen et al. 2017)
# https://github.com/shentianxiao/language-style-transfer
def load_yelp():
    filename = "language-style-transfer/data/yelp"
    dataset_name = "yelp_posneg"
    git_link = "https://github.com/shentianxiao/language-style-transfer"
    return meta_load(filename, dataset_name, git_link)


### Sentiment Positive↔Negative Amazon15 (He and McAuley 2016)
# https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data/amazon
def load_amazon(): 
    filename = "Sentiment-and-Style-Transfer/data/amazon"
    dataset_name = "amazon_posneg"
    git_link = "https://github.com/lijuncen/Sentiment-and-Style-Transfer/"
    return meta_load(filename, dataset_name, git_link)

### Used for multiple datasets with the same structure. atm: Yelp, Amazon 
def meta_load(fn, dsn, git):
    filename = fn
    dataset_name = dsn

    dataset_path = F"{path_datasets}/{filename}/"
    dataframe_path = F"{path_dataframes}/{dataset_name}"
    
    if os.path.exists(dataframe_path): # if dataframe has been created before
        df = pd.read_pickle(dataframe_path)
    elif not os.path.exists(dataset_path): # if there is no dataframe and also no source data
        if load_git: 
            logging.warning("Git repository not available. Load from git.")
            os.system(F"git clone {git} {path_datasets}")
        else:
            logging.error(F'Dataset could not be found at {dataset_path}. Please download the data from the following repository to your local device. \nMake sure the stated path in dataset_loader.py is correct. Link: https://github.com/shentianxiao/language-style-transfer')    
            return False
    else: # if there is no dataframe but the source data could be found correctly                
        dataset_files = glob.glob(F"{dataset_path}*")
        yelp_dict = { # also valid for other datasets as yelp
            'sentiment.dev.0': ["dev", 0],
            'sentiment.dev.1': ["dev", 1],
            'sentiment.test.0': ["test", 0],
            'sentiment.test.1': ["test", 1],
            'sentiment.train.0': ["train", 0],
            'sentiment.train.1': ["train", 1]
        }
        labels = []
        dataset = []
        sentences = []
        for elem in tqdm(dataset_files, desc="Preparing dataset files and saving as pkl"): 
            
            file = elem.replace(dataset_path, "")
            if not "reference" in file:
                for l in open(elem):
                    l = l.replace("\n", "") # remove new line
                    labels.append(yelp_dict[file][1])
                    dataset.append(yelp_dict[file][0])
                    sentences.append(l)
                #print(F"{file}: {l}")
        df = pd.DataFrame({"dataset":dataset,"sentiment": labels, "sample":sentences})
        df = df.drop_duplicates(subset=["sample"], keep="first")    # sentences should be unique
        df.to_pickle(dataframe_path)
        sysprint(dsn, F"Saved created dataframe to {dataframe_path}")
    p_text = F'loaded {dsn} dataset as pandas dataframe. This is a two-class dataset (pos, neg) with uncorrelated data. It comprises N={df.shape[0]} samples. Information per sample are {df.columns}.'
    sysprint(dsn, p_text)
    return df


#=================================================================#
### Go Emotions 
### https://huggingface.co/datasets/go_emotions
def load_goemo(only_ekman=True):
    dataset_name = "goemotions_old"
    dataframe_path = F"{path_dataframes}/{dataset_name}"

    if os.path.exists(dataframe_path): # if dataframe has been created before
        df = pd.read_pickle(dataframe_path)
    else: # if there is no dataframe but the source data could be found correctly                
        datasets = load_dataset('go_emotions')
        # 43.410 train , 5426 val , and 5427 test samples
        df = pd.concat([datasets['train'].to_pandas(), datasets['test'].to_pandas(), datasets['validation'].to_pandas()], axis=0)
        df.to_pickle(dataframe_path)
    p_text = F'loaded dataset as pandas dataframe. This is a emotion dataset with 54.263 labels and 27 + Neutral labels. It comprises N={df.shape[0]} samples. Information per sample are {df.columns}.'
    
    if only_ekman is True:
        # get only the ekman emotions
        df = goemo_get_only_ekman(df)
    
    sysprint("GoEmotions", p_text)
    return df


#=================================================================#
### Shakespeare
### https://github.com/google-research/google-research/tree/master/goemotions
# wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
# wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
# wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv

def load_shakespeare():
    dataset_name = "shakespeare"
    git_link = "https://github.com/harsh19/Shakespearizing-Modern-English.git"
    filename = "Shakespeare/Shakespearizing-Modern-English/data"

    dataframe_path = F"{path_dataframes}/{dataset_name}"
    dataset_path = F"{path_datasets}/{filename}/"

    if os.path.exists(dataframe_path): # if dataframe has been created before
        df = pd.read_pickle(dataframe_path)
    elif not os.path.exists(dataset_path): # if there is no dataframe and also no source data
        if load_git: 
            logging.warning("Git repository not available. Load from git.")
            os.system(F"git clone {git_link} {path_datasets}")
        else:
            logging.error(F'Dataset could not be found at {dataset_path}. Please download the data from the following repository to your local device. \nMake sure the stated path in dataset_loader.py is correct. Link: https://github.com/shentianxiao/language-style-transfer')    
            return False
    else: # if there is no dataframe but the source data could be found correctly                
        structure = [
            [F"{dataset_path}test.modern.nltktok", "test", "1"],
            [F"{dataset_path}test.original.nltktok", "test", "0"],
            [F"{dataset_path}train.modern.nltktok", "train", "1"],
            [F"{dataset_path}train.original.nltktok", "train", "0"],
            [F"{dataset_path}valid.modern.nltktok", "valid", "1"],
            [F"{dataset_path}valid.original.nltktok", "valid", "0"]
        ]

        dataset = []
        sentiment = []
        sample = []

        for elem in structure:
            with open(elem[0], 'r') as file:
                for l in file: 
                    l = l.replace("\n", "") # remove new line
                    dataset.append(elem[1])
                    sentiment.append(elem[2])
                    sample.append(l)

        df = pd.DataFrame({"dataset": dataset, "sentiment": sentiment, "sample": sample})
        df = df.drop_duplicates(subset=["sample"], keep="first")    # sentences should be unique
        df.to_pickle(dataframe_path)
    p_text = F'loaded dataset as pandas dataframe. It is a paired shakespeare dataset, comparing original texts (label 0) with the corresponding modern translations (label 1). It comprises N={df.shape[0]} samples. Information per sample are {df.columns}.'
    sysprint("Shakespeare", p_text)
    return df
