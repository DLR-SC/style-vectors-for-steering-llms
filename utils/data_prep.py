import pandas as pd 

def goemo_get_only_ekman(df):
    """
    For GoEmotions we only want the 6 base emotions and only samples that can be unambiguous assigned to a single emotion. Therefore, we filter out all other emotions. See Sec. 4.1 
    """
    num_train_samples = 43410
    num_test_samples = 5426
    num_val_samples = 5427
    
    train_df = df.iloc[0:num_train_samples]
    test_df = df.iloc[num_train_samples:num_train_samples+num_test_samples]
    val_df = df.iloc[num_train_samples+num_test_samples:num_train_samples+num_test_samples+num_val_samples]
    all_dfs = [train_df,test_df,val_df]
    
    base_emotions = ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust']
    dataset_emotions = ['admiration',
        'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
        'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
        'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral']
    base_emotion_orig_index = [dataset_emotions.index(element) for element in base_emotions]

    selected_training_samples_single_emotions = []
    for df_tmp in all_dfs:
        for idx, sample in df_tmp.iterrows():
            # we only want samples whose emotions can be unambiguous assigned to a single emotion
            if len(sample['labels']) > 1:
                continue
            
            if any(num in sample['labels'] for num in base_emotion_orig_index):
                selected_training_samples_single_emotions.append(sample)
        
        # print(len(selected_training_samples_single_emotions))

    # the first 4343 are train set, the next 554 ones from test set and the rest is val set
    only_ekman_df = pd.DataFrame(selected_training_samples_single_emotions, columns=['text', 'labels', 'id']) # 5k samples
    
    print("The GoEmo dataset has been filtered to contain only the basic ekman emotions!")
    
    return only_ekman_df
