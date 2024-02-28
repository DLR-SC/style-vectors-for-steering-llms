## Shakespeare Classifier for Evaluation

### Dataset
the dataset is created from the existing dataset [https://github.com/harsh19/Shakespearizing-Modern-English](https://github.com/harsh19/Shakespearizing-Modern-English).

For this classifier, we use only the ´test´ and `validation` set. We then re-split them into:

- `training` (3859),
- `test` (1072) and
- `validation` (429).

The dataset is already loaded to [huggingface](https://huggingface.co/datasets/notaphoenix/shakespeare_dataset), so you can load it as follows:

```python
    from datasets import load_dataset

    load_dataset("notaphoenix/shakespeare_dataset")
```

### Training

1. Check the experiment config params on the ´experiments_config.py´
    - Add new **key** and value with config
2. run the script by providing the key

     nohup python3 shakespeare_classifier/script_trainer.py -k **key** > logs/{_filename_}.log 2>&1