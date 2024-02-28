import os
import transformers
from utils.steering_layer import SteeringLayer
from dotenv import load_dotenv

# load environment variables
load_dotenv()

def load_llm_model(device):
    """Load the LLM model and put it on the device.

    :return : llm_model, tokenizer
    """
    # load the LLM
    ALPACA_WEIGHTS_FOLDER = os.getenv('ALPACA_WEIGHTS_FOLDER')
    llm_model = transformers.AutoModelForCausalLM.from_pretrained(ALPACA_WEIGHTS_FOLDER).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(ALPACA_WEIGHTS_FOLDER)
    
    return llm_model, tokenizer

def add_steering_layers(llm_model, insertion_layers):
    """Add steering layers to the model.
    
    :return: llm_model
    """
    # create the steering layers
    for layer in insertion_layers:
        llm_model.model.layers[layer].mlp = SteeringLayer(llm_model.model.layers[layer].mlp)
    
    return llm_model


def load_llm_model_with_insertions(device, insertion_layers):
    """Load the llm model, the tokenizer and insert the steering layers into the model.

    :return: llm_model, tokenizer
    """
    
    llm_model, tokenizer = load_llm_model(device)
    llm_model = add_steering_layers(llm_model, insertion_layers)
    
    return llm_model, tokenizer
    