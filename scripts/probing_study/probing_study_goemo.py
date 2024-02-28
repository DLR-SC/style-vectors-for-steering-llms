import os
os.environ["OMP_NUM_THREADS"]="10" # has to be done before any package is imported

import glob
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils.dataset_loader as dsl
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from utils.steering_vector_loader import load_activations_goemo

ROC_IMAGE_PATH = "images"
Path(ROC_IMAGE_PATH).mkdir(parents=True, exist_ok=True) # for the ROC images

DATASET="GoEmo"
print("## USING THE GOEMOTION DATASET ##")

# VECTOR_TYPE = "training_based"
VECTOR_TYPE = "activations"

# For a fair comparison of the ROC curves between the activation and the steering vectors we need to only use the activation vectors, where we have found steering vectors
COMPARISON_TYPE = "all"
# COMPARISON_TYPE = "fair"

ACTIVATION_VECTOR_PATH = os.getenv("ACTIVATIONS_PATH_GoEmo")

TRAINED_STEERING_VECTOR_PATH = os.getenv("TRAINED_VECTORS_PATH_GoEmo")
TRAINED_STEERING_VECTOR_FILES = glob.glob(f'{TRAINED_STEERING_VECTOR_PATH}/*')
TRAINED_STEERING_VEC_MIN_LOSS = 5

if VECTOR_TYPE == "training_based":
    print("## LOADING TRAINED STEERING VECTORS ##")
elif VECTOR_TYPE == "activations":
    if COMPARISON_TYPE=="fair":
        print("## LOADING ACTIVATION VECTORS in the fair setting##")
    else:        
        print("## LOADING ACTIVATION VECTORS ##")
else:
    print("Options for VECTOR_TYPE are -training_based- or -activations-")
    exit(-1)

### LOADING ACTIVATION VECTORS for train and test set
go_emo_train, go_emo_test = load_activations_goemo(ACTIVATION_VECTOR_PATH)

# we dont have activations for all entries
go_emo_train = [entry for entry in go_emo_train if len(entry) == 3]
go_emo_test = [entry for entry in go_emo_test if len(entry) == 3]

go_emo_train_tmp = np.array(go_emo_train, dtype = object)
go_emo_train_tmp_dic = list(go_emo_train_tmp[:,1])
df_train_tmp = pd.DataFrame(go_emo_train_tmp_dic, columns =["text", "labels", "id"])

go_emo_test_tmp = np.array(go_emo_test, dtype = object)
go_emo_test_tmp_dic = list(go_emo_test_tmp[:,1])
df_test_tmp = pd.DataFrame(go_emo_test_tmp_dic, columns =["text", "labels", "id"])

### LOADING TRAINED STEERING VECTORS
labels =  [25, 17, 14, 2, 26, 11]
means = []
total_mean = []

df_goemo = dsl.load_goemo()
go_emo_train_steering = []
go_emo_test_steering = []

go_emo_train_actis_fair = []
go_emo_test_actis_fair = []

for file in tqdm(TRAINED_STEERING_VECTOR_FILES, desc="Loading trained steering vecs"):
    with open(file, 'rb') as f:
        a = pickle.load(f)
        
        for key, value in a.items():
            target_sentence = key
            steering_vector = value[0]
            for vec_i, vec in enumerate(steering_vector): # the vectors were saved as tensors with device=cuda. shape is 1,4096 and therefore squeeze
                steering_vector[vec_i] = steering_vector[vec_i].detach().cpu().numpy().squeeze()
            # activations = value[1]
            loss = value[2].detach().cpu().numpy().item()
            epoch = value[3]
            # gen_text = value[4]
            label = value[5]
            
            dsl_entry = df_goemo[df_goemo["text"] == target_sentence]
            

            if loss < TRAINED_STEERING_VEC_MIN_LOSS:

                if not (df_train_tmp[df_train_tmp["text"] == target_sentence]).empty:  
                    found = df_train_tmp[df_train_tmp["text"] == target_sentence]           
                    go_emo_train_actis_fair.append(go_emo_train[found.index[0]])   
                    go_emo_train_steering.append([label.item(), dsl_entry.to_dict(orient="list"), steering_vector, loss]) 

                elif not (df_test_tmp[df_test_tmp["text"] == target_sentence]).empty: 
                    found = df_test_tmp[df_test_tmp["text"] == target_sentence]           
                    go_emo_test_actis_fair.append(go_emo_test[found.index[0]])   
                    go_emo_test_steering.append([label.item(), dsl_entry.to_dict(orient="list"), steering_vector, loss])     

### TRAINED STEERING VECTORS ARE LOADED


# taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_classification(y_train,y_test,y_score, n_classes, target_names, layer_indices):
    from itertools import cycle
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import RocCurveDisplay

    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_test.shape  # (n_samples, n_classes)
    fig, ax = plt.subplots(figsize=(6, 6))

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        #label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        label=f"micro-average (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "purple", "green"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            #name=f"ROC curve for {target_names[class_id]}",
            name=f"{target_names[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--") #, label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.grid(color='lightgray', linestyle='-', linewidth=1)
    plt.xlabel("False Positive Rate", fontsize = 15)
    plt.ylabel("True Positive Rate", fontsize = 15)
    # plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend(loc = "lower right", fontsize = 13)
    fig_name_indices = ""
    for layer_idx in layer_indices:
        fig_name_indices += f"{layer_idx}_"
    fig_name = f"ROC_goemo_{fig_name_indices}steering.pdf" if VECTOR_TYPE == "training_based" else f"ROC_goemo_{fig_name_indices}actis_{COMPARISON_TYPE}.pdf"
    # fig_name = f"test.pdf"
    plt.savefig(f"{ROC_IMAGE_PATH}/{fig_name}")
    plt.clf()

# logistic regression iterating over all layers
def single_layer_classification(layer_index):
    """Training a logistic regression classifier with single layers as input. 

    :param int layer_index: Which layer should be used.
    """
    
    if (VECTOR_TYPE == "activations") and (COMPARISON_TYPE == "all"):
        Y_train = []
        X_train = []
        for entry in go_emo_train:
            Y_train.append(labels.index(entry[1]['labels'][0]))
            X_train.append(entry[2][layer_index])

        Y_test = []
        X_test = []
        for entry in go_emo_test:
            Y_test.append(labels.index(entry[1]['labels'][0]))
            X_test.append(entry[2][layer_index])

    else:
        Y_25, Y_17, Y_14, Y_2, Y_26, Y_11 = [],[],[],[],[],[]
        X_25, X_17, X_14, X_2, X_26, X_11 = [],[],[],[],[],[]
        
        entry_list = go_emo_train_steering if VECTOR_TYPE == "training_based" else go_emo_train_actis_fair

        for entry in entry_list:
            class_label = entry[1]['labels'][0]

            if class_label == 25:
                Y_25.append(labels.index(entry[1]['labels'][0]))
                X_25.append(entry[2][layer_index-18])
            elif class_label == 17:
                Y_17.append(labels.index(entry[1]['labels'][0]))
                X_17.append(entry[2][layer_index-18])
            elif class_label == 14:
                Y_14.append(labels.index(entry[1]['labels'][0]))
                X_14.append(entry[2][layer_index-18])
            elif class_label == 2:
                Y_2.append(labels.index(entry[1]['labels'][0]))
                X_2.append(entry[2][layer_index-18])
            elif class_label == 26:
                Y_26.append(labels.index(entry[1]['labels'][0]))
                X_26.append(entry[2][layer_index-18])
            elif class_label == 11:
                Y_11.append(labels.index(entry[1]['labels'][0]))
                X_11.append(entry[2][layer_index-18])
            else:
                print(f"Didn't find {class_label}")

        X_train, X_test = [],[]
        Y_train, Y_test = [], []
        split_ratio = .5
        for tup in [(X_25,Y_25), (X_17,Y_17), (X_14,Y_14), (X_2,Y_2), (X_26,Y_26), (X_11,Y_11)]:
            end_train_idx = int(split_ratio * len(tup[0]))+1
            X_train.extend(tup[0][0:end_train_idx])
            Y_train.extend(tup[1][0:end_train_idx])
            X_test.extend(tup[0][end_train_idx:-1])
            Y_test.extend(tup[1][end_train_idx:-1])
                

    clf = LogisticRegression(multi_class='multinomial', max_iter = 20000, class_weight='balanced').fit(X_train, Y_train)
    print(f"Layer {layer_index} classification score: {clf.score(X_test,Y_test)}")
    plot_classification(Y_train,Y_test, clf.predict_proba(X_test), 6, ["sadness", "joy", "fear", "anger", "surprise", "disgust"], [layer_index])


# logistic regression with concatenated layers, sliding window
def multi_layer_classification(num_layers = 3, specific_layers = None):
    """Training a logistic regression classifier with multiple layers as input. 
    Currently it only works for the activation-based vectors.

    :param int num_layers: How many layers per classifier, defaults to 3
    :param array specific_layers: Which layers should be used , defaults to None
    """
    
    layer_indices_list = []
    if specific_layers is not None:
        layer_indices_list = [specific_layers] 
    else:
        for i in range(0,33):
            layer_indices_list.append(np.arange(i,i+num_layers))

    
    for layer_indices in layer_indices_list:

        Y_train = []
        X_train = []
        for entry in go_emo_train:
            Y_train.append(labels.index(entry[1]['labels'][0]))
            entries = []
            for layer_index in layer_indices:
                entries.append(entry[2][layer_index])
            X_train.append(np.concatenate(entries))

        Y_test = []
        X_test = []
        for entry in go_emo_test:
            Y_test.append(labels.index(entry[1]['labels'][0]))
            entries = []
            for layer_index in layer_indices:
                entries.append(entry[2][layer_index])
            X_test.append(np.concatenate(entries))

        clf = LogisticRegression(multi_class='multinomial', max_iter = 10000, class_weight='balanced').fit(X_train, Y_train)
        print(f"Layer {layer_indices[0]} classification score: {clf.score(X_test,Y_test)}")
        plot_classification(Y_train,Y_test, clf.predict_proba(X_test), 6, ["sadness", "joy", "fear", "anger", "surprise", "disgust"], layer_indices)


# layers = [18,19,20] # commented for tests
layers = [18]
for layer in layers:
    single_layer_classification(layer)
