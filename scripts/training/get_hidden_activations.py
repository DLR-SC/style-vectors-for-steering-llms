import os
import torch
import transformers
from utils import dataset_loader as dsl
from tqdm import tqdm
import pickle
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

MODEL_PATH = os.getenv("ALPACA_WEIGHTS_FOLDER")

# DATASET = "yelp"
DATASET = "GoEmo"
# DATASET = "shakes"

PATH_TO_ACTIVATION_STORAGE = ""
if DATASET=="yelp":
    PATH_TO_ACTIVATION_STORAGE = os.getenv("ACTIVATIONS_PATH_YELP")
if DATASET=="GoEmo":
    PATH_TO_ACTIVATION_STORAGE = os.getenv("ACTIVATIONS_PATH_GoEmo")
if DATASET=="shakes":
    PATH_TO_ACTIVATION_STORAGE = os.getenv("ACTIVATIONS_PATH_Shake")
Path(PATH_TO_ACTIVATION_STORAGE).mkdir(parents=True, exist_ok=True) 

# select which device you want to use
DEVICE = torch.device('cuda:1')
# create the model 
alpaca_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
alpaca_model.to(DEVICE)

# this function extracts the activation vectors for every layer during the forward pass of a prompt and saves them as .pkl files
def process_dataset(dataset_name):
    df = []
    print(f"Saving activations for {dataset_name} dataset!")
    if "shakes" in dataset_name:
        df = dsl.load_shakespeare()
    elif "GoEmo" in dataset_name:
        df = dsl.load_goemo()
    elif "yelp" in dataset_name:
        df = dsl.load_yelp()
    else:
        print(f"Didnt recognize {dataset_name}!")
        exit(-1)
    actis, actis_train, actis_test = [],[],[]
    i = 0
    j = 0
    for index, row in tqdm(df.iterrows(), desc=f"Iterating through all samples from the {DATASET} dataset and extracting the activations", total=df.shape[0]):

        # removing newlines from samples.
        sentence = []
        if "GoEmo" in dataset_name:
            sentence = row['text'].replace('\n', '')
        else:
            sentence = row['sample'].replace('\n', '')
        input_tokens = alpaca_tokenizer(sentence, return_tensors="pt").to(DEVICE)

        # skip samples with more than 300 tokens, otherwise GPU runs out of memory
        if len(input_tokens.input_ids) > 300: 
            continue
        gen_text = alpaca_model.forward(input_tokens.input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = []

        #iterating over all layers and storing activations of the last token
        for layer in gen_text['hidden_states']:
            hidden_states.append(layer[0][-1].detach().cpu().numpy())

        if DATASET == "GoEmo":
            actis.append([index,row,hidden_states])
        else:
            # shakespeare and yelp store the labels in column 'sentiment', go emotion stores labels in 'labels' column.
            actis.append([index, sentence, hidden_states, row['sentiment']])

            i += 1
            # save activations in batches
            if i == 10000:
                i = 0
                with open(f'{PATH_TO_ACTIVATION_STORAGE}/{dataset_name}_activations_{j}.pkl', 'wb') as f:
                    pickle.dump(actis, f)
                del actis
                del hidden_states
                actis = []
                j += 1
    
    if DATASET=="GoEmo":
        actis_train = actis[0:4343] # training set
        actis_test = actis[4343:4343+554] # test set
        # we ignore the val set
        with open(f'{PATH_TO_ACTIVATION_STORAGE}/{dataset_name}_activations_train.pkl', 'wb') as f:
                pickle.dump(actis_train, f)
        with open(f'{PATH_TO_ACTIVATION_STORAGE}/{dataset_name}_activations_test.pkl', 'wb') as f:
                pickle.dump(actis_test, f)
    else:    
        # in case the number of samples is not dividable by 10000, we safe the rest
        with open(f'{PATH_TO_ACTIVATION_STORAGE}/{dataset_name}_activations_{j}.pkl', 'wb') as f:
                pickle.dump(actis, f)
    
    del actis
    del hidden_states
    
# call the function to extract the activations during the forward pass
process_dataset(DATASET)
