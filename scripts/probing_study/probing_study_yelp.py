import os
os.environ["OMP_NUM_THREADS"]="10" # has to be done before any package is imported
import glob
import pickle
import numpy as np
import random
import utils.dataset_loader as dsl
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
ROC_IMAGE_PATH = "images"
Path(ROC_IMAGE_PATH).mkdir(parents=True, exist_ok=True) # for the ROC images

DATASET="yelp"
print("## USING THE YELP DATASET ##")

# decision which vector type should be loaded
# VECTOR_TYPE = "training_based"
VECTOR_TYPE = "activations"

# For a fair comparison of the ROC curves between the activation and the steering vectors we need to only use the activation vectors, where we have found steering vectors
# COMPARISON_TYPE = "fair"
COMPARISON_TYPE = "all"  # use all activation vectors

ACTIVATIONS_VECTOR_PATH = os.getenv("ACTIVATIONS_PATH_YELP")
ACTIVATIONS_VECTOR_FILES = glob.glob(f'{ACTIVATIONS_VECTOR_PATH}/{DATASET}*') + glob.glob(f'{ACTIVATIONS_VECTOR_PATH}/{DATASET.title()}*') # .title() because we saved the yelp files as Yelp...

TRAINED_STEERING_VECTOR_PATH = os.getenv("TRAINED_VECTORS_PATH_Yelp")
TRAINED_STEERING_VECTOR_FILES = glob.glob(f'{TRAINED_STEERING_VECTOR_PATH}/*')
TRAINED_STEERING_VEC_MIN_LOSS = 5

if VECTOR_TYPE == "training_based":
    print("## LOADING STEERING VECTORS ##")
elif VECTOR_TYPE == "activations":
    print("## LOADING ACTIVATION VECTORS ##")    
else:
    print("Options for VECTOR_TYPE are -training_based- or -activations-")
    exit(-1)

df_yelp = dsl.load_yelp()

positive, positive_acti = [], []
negative, negative_acti = [], []

# if VECTOR_TYPE == "training_based":
# always load the steering vectors. We need them for a fair comparison with the activations
for file in tqdm(TRAINED_STEERING_VECTOR_FILES, desc="Loading trained steering vecs"):
    with open(file, 'rb') as f:
        a = pickle.load(f)
        for key, value in a.items():
            target_sentence = key
            steering_vector = value[0]
            for vec_i, vec in enumerate(steering_vector): # the vectors were saved as tensors with device=cuda. shape is 1,4096 and therefore squeeze
                steering_vector[vec_i] = steering_vector[vec_i].detach().cpu().numpy().squeeze()
            activations = value[1]
            loss = value[2].detach().cpu().numpy().item()
            epoch = value[3]
            gen_text = value[4]
            label = value[5]

            if loss < TRAINED_STEERING_VEC_MIN_LOSS:
                if label:
                    positive.append([steering_vector, target_sentence, loss, label])
                else:
                    negative.append([steering_vector, target_sentence, loss, label])                     

print(f"Number of positive trained steering samples with loss < {TRAINED_STEERING_VEC_MIN_LOSS}: {len(positive)}")
print(f"Number of negative trained steering samples with loss < {TRAINED_STEERING_VEC_MIN_LOSS}: {len(negative)}")

if VECTOR_TYPE == "activations":
    positive = np.asarray(positive, dtype = "object")
    negative = np.asarray(negative, dtype = "object")
    idx = 0
    for file in tqdm(ACTIVATIONS_VECTOR_FILES, desc="Loading activations"):
        if (idx == 4) and (COMPARISON_TYPE == "all"): # we can't load all activations vectors due to memory constraints
            break
        with open(file, 'rb') as f:
            a = pickle.load(f)
            for entry in a:
                df_entry = df_yelp.loc[entry[0]]
                label = df_entry['sentiment']
                target_sentence = entry[1]
                activation_vectors = entry[2] # list of 33 vectors. Each vector has 4096 values

                if COMPARISON_TYPE == "fair":
                    steering_vec_exists = (target_sentence in positive[:,1]) or (target_sentence in negative[:,1])
                    if not steering_vec_exists:
                        continue

                if label:
                    positive_acti.append([activation_vectors, target_sentence, label])
                else:
                    negative_acti.append([activation_vectors, target_sentence, label])
        idx += 1
    
    if COMPARISON_TYPE == "fair":
        positive = positive_acti
        negative = negative_acti
    if COMPARISON_TYPE == "all":
        positive = random.sample(positive_acti,10000)
        negative = random.sample(negative_acti,10000)

    print(f"Number of activation-based positive samples: {len(positive)}")
    print(f"Number of activation-based negative samples: {len(negative)}")

split_ratio = 0.5
training_set_size = int(split_ratio * len(positive))
train_positive = positive[:training_set_size]
test_positive = positive[training_set_size:]

training_set_size = int(split_ratio * len(negative))
train_negative = negative[:training_set_size]
test_negative = negative[training_set_size:]

X, y = [], []
fpr_list, tpr_list, roc_auc_list = [],[],[]

if VECTOR_TYPE == "training_based":
    # chosen_indices = [0,1,2] # commented for tests
    # description_indices = [18,19,20] # commented for tests
    chosen_indices = [0]
    description_indices = [18]
elif COMPARISON_TYPE == "fair":
    # chosen_indices = [18,19,20] # commented for tests
    # description_indices = [18,19,20] # commented for tests
    chosen_indices = [18]
    description_indices = [18]
elif VECTOR_TYPE == "activations":
    # we only want to plot chosen layers, so that the plot isn't cluttered
    all_indices = range(33)
    # chosen_indices = [0,1,2,3,5,10,15,18,19,20,25,30] # commented for tests
    chosen_indices = [15, 16]
    description_indices = chosen_indices

# train/load the classifier for the vectors
for i in tqdm(chosen_indices, desc="Calculating ROC per chosen layer"):

    for n in train_positive:
        X.append(n[0][i])
        y.append(1)
    for n in train_negative:
        X.append(n[0][i])
        y.append(0)

    clf = LogisticRegression(random_state=0, max_iter = 1000).fit(X, y)

    Ts = 0
    Fs = 0
    preds = []
    test_y = []

    for n in test_positive:
        pred = clf.predict_proba([n[0][i]])[0]
        cls = np.argmax(pred)
        if cls:
            Ts += 1
        else:
            Fs += 1
        preds.append(pred)
        test_y.append(1)

    for n in test_negative:
        pred = clf.predict_proba([n[0][i]])[0]
        cls = np.argmax(pred)
        if not cls:
            Ts += 1
        else:
            Fs += 1
        preds.append(pred)
        test_y.append(0)

    # print(clf.score(test_y))

    fpr, tpr, thresholds = metrics.roc_curve(test_y, [p[1] for p in preds])
    roc_auc = metrics.auc(fpr, tpr)

    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)

towards_red_cmap = plt.cm.get_cmap('YlOrRd') # choose colormap
if (VECTOR_TYPE == "training_based") or (COMPARISON_TYPE=="fair"):
    min_cmap_value = 0.89 # except for one value, the values presented in the figure are above 0.9. Therefore we want to scale the colormap accordingly  
elif VECTOR_TYPE == "activations":
    min_cmap_value = 0.93

# To get an expressive color at the top range of values we utilize exponential scaling
color_value_rescaled_list = [np.exp(((roc_auc - min_cmap_value) / (1.-min_cmap_value) * 1. if roc_auc > min_cmap_value else 0.2))-1 for roc_auc in roc_auc_list]
max_color_value_rescaled = np.max(color_value_rescaled_list)

fig, ax = plt.subplots(figsize=(6, 6))
for i,_ in tqdm(enumerate(fpr_list), desc="Plotting ROC curves"):
    roc_auc = roc_auc_list[i]
    
    # color values have to be between 0 and 1 for the cmap. Therefore, we scale it accordingly
    color_value_rescaled = color_value_rescaled_list[i]/max_color_value_rescaled

    title = f'Receiver Operating Characteristic - Yelp / {VECTOR_TYPE}' if (VECTOR_TYPE == "training_based") else f'Receiver Operating Characteristic - Yelp / {VECTOR_TYPE} / {COMPARISON_TYPE}'
    # plt.title(title)
    plt.axis("square")
    plt.plot(fpr_list[i], tpr_list[i], label = f'Layer {description_indices[i]} / AUC={roc_auc:.2f}', color = towards_red_cmap(color_value_rescaled))
    plt.legend(loc = 'lower right', fontsize = 13)
    plt.grid(color='lightgray', linestyle='-', linewidth=1)
    plt.plot([0, 1], [0, 1],'k--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize = 15)
    plt.xlabel('False Positive Rate', fontsize = 15)
    img_name = f"ROC_yelp_{VECTOR_TYPE}.pdf" if (VECTOR_TYPE == "training_based") else f"ROC_yelp_{VECTOR_TYPE}_{COMPARISON_TYPE}.pdf"
    plt.savefig(f"{ROC_IMAGE_PATH}/{img_name}", format="pdf")

    # print(f"Trainingdata Statistics: Positive Training Samples: {len(train_positive)} Negative Training Samples: {len(train_negative)}")
    # print(f"Testdata Statsitics: Positive Test Samples: {len(test_positive)} Negative Test Samples: {len(test_negative)}")
    # print(f"Number of correct classified  sentences: {Ts}")
    # print(f"Number of incorrect classified  sentences: {Fs}")
    # print(f"Percentage of correct classifications: {Ts / (Ts+Fs)}\n")
    #####################################e################################################