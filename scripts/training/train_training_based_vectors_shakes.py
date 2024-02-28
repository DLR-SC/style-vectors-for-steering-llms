import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from transformers import set_seed
import numpy as np
from torch import nn, optim
import pickle
import nltk
from utils import dataset_loader as dsl
from utils.steering_layer import SteeringLayer
from torch import cuda
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
DEVICE = torch.device('cuda:0')

SEED = 1337
set_seed(SEED)
torch.set_default_dtype(torch.float32)

INSERTION_LAYERS = [18, 19, 20] # best layers in our experiments
MODEL_PATH = os.getenv("ALPACA_WEIGHTS_FOLDER")
TRAINED_STEERING_VECTOR_PATH = os.getenv("TRAINED_VECTORS_PATH_Shake")

Path(TRAINED_STEERING_VECTOR_PATH).mkdir(parents=True, exist_ok=True)
    
MAX_NEW_TOKENS = 20
TARGET_SENTENCE_COUNTER = 0

# helper function to save a trained steering vector
def save_trained_steering_vector(steering_vector, target_sentence, final_loss, layer_of_interest, epoch_of_extraction, gen_text, label, TARGET_SENTENCE_COUNTER):
    save_dict = {target_sentence: (steering_vector, layer_of_interest, final_loss, epoch_of_extraction, gen_text, label)}

    with open(f"{TRAINED_STEERING_VECTOR_PATH}/LLMB{str(INSERTION_LAYERS)}_{str(TARGET_SENTENCE_COUNTER)}.pkl", 'wb') as fp:
        pickle.dump(save_dict, fp)
        print(f"Steering vector for sentence \"{target_sentence}\" saved at {TRAINED_STEERING_VECTOR_PATH}/LLMB{str(INSERTION_LAYERS)}_{str(TARGET_SENTENCE_COUNTER)}.pkl")


df_shake = dsl.load_shakespeare()
# we filter out samples with a length over 50 due to time constraints. See Sec. 4.2 in our publication for more information
df = df_shake[df_shake['sample'].str.len() < 50] 
df = df.sample(n=6000)
# print("Shakespeare Dataset loaded")

# BOS TOKEN ID: 1
# EOS TOKEN ID: 2
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
# print("Tokenizer loaded")

model = LlamaForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)
# print("Model loaded")

model.to(DEVICE)
# print("model and device on device")
# print(model)

# Only compute gradients for steering vector
for param in model.parameters():
    param.requires_grad=False
# Adding our custom steering layer to the model
for insert_layer in INSERTION_LAYERS:
    model.model.layers[insert_layer].mlp = SteeringLayer(model.model.layers[insert_layer].mlp)
# model.model.layers[INSERTION_LAYER].self_attn = CustomSteerLayer(model.model.layers[INSERTION_LAYER].self_attn)

# corpus = ["Hello, world!", # steering vector training takes about 35-40 epochs
#         "All your base are belong to us", # this takes considerably longer.
#         "The quick brown fox jumps over the lazy dog"]  # this takes considerably longer.

EPOCHS = 400
learning_rate = 0.01
decayRate = 0.96
num_tokens_to_predict = 50
current_lr = learning_rate
for index, row in df.iterrows():
    label = row["sentiment"]
    target = row["sample"] # row[5]

    # Get raw activations for target sentence
    for insert_layer in INSERTION_LAYERS:
        model.model.layers[insert_layer].mlp.add_steering = False
    target_tokens = tokenizer(target, return_tensors="pt").to(DEVICE)
    model_output = model.forward(target_tokens.input_ids)
    raw_activations = []
    for insert_layer in INSERTION_LAYERS:
        # model.model.layers[insert_layer].mlp.activations
        model.model.layers[insert_layer].mlp.add_steering = True

    # Init steering vector
    for insert_layer in INSERTION_LAYERS:
        model.model.layers[insert_layer].mlp.reset_steering_vector()
        print(f"Initial Steering Vector: {model.model.layers[insert_layer].mlp.steering_vector}")
    # model.model.layers[INSERTION_LAYER].mlp.add_steering = True

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    custom_layers = []
    for insert_layer in INSERTION_LAYERS:
        custom_layers.append(model.model.layers[insert_layer].mlp.steering_vector)
    adam_optim = optim.Adam(custom_layers, lr=learning_rate)
    # adam_optim = optim.Adam([model.model.layers[INSERTION_LAYER].mlp.steering_vector], lr=learning_rate)
    
    
    current_bleu = 0
    epoch_of_extraction = 0
    for current_epoch in range(0,EPOCHS):
        if current_epoch >= 1: current_bleu = BLEUscore
        overall_loss = 0
        target_tokens = tokenizer(target, return_tensors="pt").to(DEVICE)
        input_tokens = tokenizer("", return_tensors="pt").to(DEVICE)
        gen_tokens = []

        for j in range(len(target_tokens.input_ids[0])-1):
            if j == 0:
                model_output = model.forward(input_tokens.input_ids)
                logits = model_output.logits
                gen_tokens.append(np.argmax(logits.detach().cpu()))
                past_key_vals = model_output.past_key_values
                overall_loss += loss_fn(logits[0][0], target_tokens.input_ids[0][j+1])
            else:
                model_output = model.forward(torch.Tensor([[np.argmax(logits.detach().cpu())]]).type(torch.int64).to(DEVICE), past_key_values = past_key_vals)
                logits = model_output.logits
                gen_tokens.append(np.argmax(logits.detach().cpu()))
                past_key_vals = model_output.past_key_values
                overall_loss += loss_fn(logits[0][0], target_tokens.input_ids[0][j+1])


        BLEUscore = nltk.translate.bleu_score.sentence_bleu([target.split()], " ".join(tokenizer.batch_decode(gen_tokens)).split())

        overall_loss.backward()
        adam_optim.step()
        
        if overall_loss < 100:
            for g in adam_optim.param_groups:
                g["lr"] = 0.01
                current_lr = g["lr"]
                
        epoch_of_extraction = current_epoch
        final_loss = overall_loss
        if gen_tokens == [b for b in target_tokens.input_ids[0][1:]]:
            print("====================================================================================================")
            print("====================================================================================================")
            print("Matching steering vector found! Stopping training.")
            print(f"Final Epoch {current_epoch}\nTarget sentence: {target}")
            # print(f"Final Steering Vector Gradient: {model.model.layers[INSERTION_LAYER].mlp.steering_vector.grad}")
            print(f"Final generated text: {tokenizer.batch_decode(gen_tokens)}")
            print(f"Final BLEU score: {BLEUscore}")
            print(f"Loss: {overall_loss}")
            # print(f"Final Steering Vector: {model.model.layers[INSERTION_LAYER].mlp.steering_vector}")
            # save_trained_steering_vector(model.model.layers[INSERTION_LAYER].mlp.steering_vector, target, label)
            custom_layers = []
            for insert_layer in INSERTION_LAYERS:
                custom_layers.append(model.model.layers[insert_layer].mlp.steering_vector.data)
            save_trained_steering_vector(custom_layers, 
                                target, final_loss, raw_activations, 
                                epoch_of_extraction, " ".join(tokenizer.batch_decode(gen_tokens)), label, TARGET_SENTENCE_COUNTER)
            TARGET_SENTENCE_COUNTER += 1
            break
        elif current_epoch == (EPOCHS-1):
            print("====================================================================================================")
            print(f"Epoch {current_epoch}\nTarget sentence: {target}")
            # print(f"Current Steering Vector Gradient: {model.model.layers[INSERTION_LAYER].mlp.steering_vector.grad}")
            print(f"Current generated text: {tokenizer.batch_decode(gen_tokens)}")
            print(f"Current BLEU score: {BLEUscore}")
            print(f"Current learning rate: {current_lr}")
            print(f"Loss: {overall_loss}")
            # print(f"Updated Steering Vector: {model.model.layers[INSERTION_LAYER].mlp.steering_vector}")
            if BLEUscore > current_bleu:
                custom_layers = []
                for insert_layer in INSERTION_LAYERS:
                    custom_layers.append(model.model.layers[insert_layer].mlp.steering_vector.data)
                save_trained_steering_vector(custom_layers, 
                                    target, final_loss, raw_activations, 
                                    epoch_of_extraction, " ".join(tokenizer.batch_decode(gen_tokens)), label, TARGET_SENTENCE_COUNTER)
                TARGET_SENTENCE_COUNTER += 1
        else:
            print("====================================================================================================")
            print(f"Epoch {current_epoch}\nTarget sentence: {target}")
            # print(f"Current Steering Vector Gradient: {model.model.layers[INSERTION_LAYER].mlp.steering_vector.grad}")
            print(f"Current generated text: {tokenizer.batch_decode(gen_tokens)}")
            print(f"Current BLEU score: {BLEUscore}")
            print(f"Current learning rate: {current_lr}")
            print(f"Loss: {overall_loss}")
            # print(f"Updated Steering Vector: {model.model.layers[INSERTION_LAYER].mlp.steering_vector}")
            if BLEUscore > current_bleu:
                custom_layers = []
                for insert_layer in INSERTION_LAYERS:
                    custom_layers.append(model.model.layers[insert_layer].mlp.steering_vector.data)
                save_trained_steering_vector(custom_layers, 
                                    target, final_loss, raw_activations, 
                                    epoch_of_extraction, " ".join(tokenizer.batch_decode(gen_tokens)), label, TARGET_SENTENCE_COUNTER)

