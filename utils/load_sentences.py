import os

def load_sentences(which_kind):
    """Load the sentences of the specified kind."""
    # get correct path
    path_to_sentences = None
    if which_kind=="subjective":
        path_to_sentences = "evaluation_prompts/subjective_sentences.txt"
    elif which_kind=="factual":
        path_to_sentences = "evaluation_prompts/factual_sentences.txt"
    # get absolute path
    path_to_sentences = os.path.join(os.getcwd(),path_to_sentences)
    print(path_to_sentences)
    # read sentences and save them
    sentences = []
    with open(path_to_sentences,'r',encoding="utf-8") as tfile:
        sentences = tfile.readlines()
    # strip away \n
    stripped = [s.strip() for s in sentences]    
    
    return stripped

def load_factual_sentences():
    """Returns the factual sentences used for evaluation of our model.

    :return list: List of factual sentences
    """
    return load_sentences(which_kind="factual")

def load_subjective_sentences():
    """Returns the subjective sentences used for evaluation of our model.

    :return list: List of subjective sentences
    """
    return load_sentences(which_kind="subjective")

def load_all_sentences():
    """Returns the factual and subjective sentences used for evaluation of our model.

    :return list, list: List of factual sentences, list of subjective sentences
    """
    return load_sentences(which_kind="factual"), load_sentences(which_kind="subjective")

if __name__ == '__main__':
    a,b = load_all_sentences()
    
    print("Factual Sentences: ", a, "\n\n")
    print("Subjective Sentences: ", b, "\n\n")