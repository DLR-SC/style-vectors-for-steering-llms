import glob
import pickle
import utils.dataset_loader as dsl
from tqdm import tqdm
import random

def load_activations_goemo(vector_path):
    with open(f'{vector_path}/GoEmo_activations_train.pkl', 'rb') as f:
        go_emo_train = pickle.load(f)

    with open(f'{vector_path}/GoEmo_activations_test.pkl', 'rb') as f:
        go_emo_test = pickle.load(f)
        
    return go_emo_train, go_emo_test

def load_trained_vectors_yelp(vector_path):

    vector_files = glob.glob(f'{vector_path}/*')

    INSERTION_LAYERS =  [x for x in vector_files if '18, 19, 20' in x]
    positive = 0
    negative = 0
    steering_vectors = []

    for file in INSERTION_LAYERS:
        with open(file, 'rb') as f:
            a = pickle.load(f)
            for key, value in a.items():
                target_sentence = key
                steering_vector = value[0]
                activations = value[1]
                loss = value[2]
                epoch = value[3]
                gen_text = value[4]
                label = value[5]

                if loss < 5:
                    steering_vectors.append([steering_vector, target_sentence, epoch, loss, gen_text, label])

                    if label:
                        positive += 1
                    #     positive.append([steering_vector, target_sentence, epoch, loss])
                    #     # positive.append([steering_vector, activations, loss, epoch, target_sentence])
                    else:
                        negative += 1
                    #     negative.append([steering_vector, target_sentence, epoch, loss])
                    #     # negative.append([steering_vector, activations, loss, epoch, target_sentence])

    print(f"Number of positive samples: {positive}")
    print(f"Number of negative samples: {negative}")
    # print(f"Number of steering vectors with training loss < 5: {len(steering_vectors)}")

    return steering_vectors



def load_activations_yelp(vector_path, dataset, num_activation_files_to_load=4):
    
    vector_files = glob.glob(f'{vector_path}/{dataset}*') + glob.glob(f'{vector_path}/{dataset.title()}*') # we stored activations as Yelp, but it should usually be yelp

    df_yelp = dsl.load_yelp()

    positive = []
    negative = []
    steering_vectors = []
    idx = 0
    for file in vector_files:
        if idx == num_activation_files_to_load: # only load a certain amount of them due to memory problems
            break
        count = 0
        with open(file, 'rb') as f:
            a = pickle.load(f)
            # print(a)
            random.shuffle(a)
            for entry in tqdm(a):
                # we need this, because our activation vectors still contained duplicates
                try:
                    df_entry = df_yelp.loc[entry[0]]
                except KeyError:
                    continue
                label = df_entry['sentiment']
                target_sentence = entry[1]
                steering_vector = entry[2]
                # activations = value[1]
                # loss = value[2]
                # epoch = value[3]
                # gen_text = value[4]
                # label = value[5]
                if label:
                    positive.append([steering_vector, target_sentence, label])
                    steering_vectors.append([steering_vector, target_sentence, label])
                    # positive.append([steering_vector, activations, loss, epoch, target_sentence])
                else:
                    negative.append([steering_vector, target_sentence, label])
                    steering_vectors.append([steering_vector, target_sentence, label])
                    # negative.append([steering_vector, activations, loss, epoch, target_sentence])
                # if count == 2000: 
                #     print()
                #     break
                count += 1

        idx += 1

    print(f"Number of positive acti vectors: {len(positive)}")
    print(f"Number of negative acti vectors: {len(negative)}")

    return steering_vectors


def load_activations_shake(vector_path, dataset):
    
    vector_files = glob.glob(f'{vector_path}/{dataset}*')

    df_shake = dsl.load_shakespeare()

    modern_shakespeare_train_acti, original_shakespeare_train_acti, modern_shakespeare_test_acti, original_shakespeare_test_acti = [],[],[],[]
    negative = []
    steering_vectors = []
    idx = 0
    print(f"Number of files: {vector_files}")
    for file in vector_files:
        with open(file, 'rb') as f:
            a = pickle.load(f)
            for entry in a:
                # we need this, because our activation vectors still contained duplicates
                try:
                    df_entry = df_shake.loc[entry[0]]
                except KeyError:
                    continue
                label = df_entry['sentiment']
                target_sentence = df_entry['sample']
                activations = entry[2]
                training_set = True if df_entry['dataset'] == 'train' else False
                test_set = True if df_entry['dataset'] == 'test' else False

                if training_set:
                    if int(label) == 1: # modern
                        modern_shakespeare_train_acti.append([activations, target_sentence, int(label)])
                        steering_vectors.append([activations, target_sentence, int(label)])
                    else: # original
                        original_shakespeare_train_acti.append([activations, target_sentence, int(label)])
                        steering_vectors.append([activations, target_sentence, int(label)])
                if test_set:
                    if int(label) == 1: # modern
                        modern_shakespeare_test_acti.append([activations, target_sentence, int(label)])
                        steering_vectors.append([activations, target_sentence, int(label)])
                    else: # original
                        original_shakespeare_test_acti.append([activations, target_sentence, int(label)])
                        steering_vectors.append([activations, target_sentence, int(label)])

    print(f"Original shake train activation vectors: {len(original_shakespeare_train_acti)}")
    print(f"Modern shake train activation vectors: {len(modern_shakespeare_train_acti)}")
    print(f"Original shake test activation vectors: {len(original_shakespeare_test_acti)}")
    print(f"Modern shake test activation vectors: {len(modern_shakespeare_test_acti)}")

    return steering_vectors