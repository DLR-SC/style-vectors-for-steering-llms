import dataclasses
from typing import ClassVar, List
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.dataset_dict import Dataset, DatasetDict
from dataclasses import field


@dataclasses.dataclass
class ShakespeareanDataset():

    splits: List = field(default_factory=lambda: [ "test", "valid"])
    data_format: str = "classification" # or parallel
    _DATA_PATH_: ClassVar = "shakespeare_classifier/original_data/{split}.{type}.nltktok"

    _SPLIT_TRAINING_: ClassVar = "train"
    _SPLIT_TEST_: ClassVar = "test"
    _SPLIT_VALIDATION_: ClassVar = "valid"

    _LABEL2ID_: ClassVar = {'modern':0, 'shakespearean': 1} 
    _ID2LABEL_: ClassVar = {0: 'modern', 1: 'shakespearean'}

    def reformat_data(self) -> pd.DataFrame:
        if self.data_format == "classification":
            original = []
            modern = []
            for split in self.splits:
                with open(ShakespeareanDataset._DATA_PATH_.format(split=split, type="original")) as f:
                    original.extend(f.readlines())#for split in self.splits}
                with open(ShakespeareanDataset._DATA_PATH_.format(split=split, type="modern")) as f:
                    modern.extend(f.readlines())#for split in self.splits}
            original_df = pd.DataFrame(columns=["text", "label"], data=list(zip(original, [ShakespeareanDataset._LABEL2ID_["shakespearean"]]*len(original))))
            shakespearean_df = pd.DataFrame(columns=["text", "label"], data=list(zip(modern, [ShakespeareanDataset._LABEL2ID_["modern"]]*len(modern))))

            _df = pd.concat([original_df, shakespearean_df])
            return _df
    
    def get_dataset(self, force_reload: bool = False) -> DatasetDict:
        
        dataset_name: str = "notaphoenix/shakespeare_dataset"

        try:
            if not force_reload:
                labelled_dataset = load_dataset(dataset_name,
                                                use_auth_token=True) 
        except FileNotFoundError:
            force_reload = True
            print("dataset was not found on hugging face.")

        if force_reload:
            print("(Re)creating huggingface dataset from local dataset")
            df: pd.DataFrame = self.reformat_data()

            X_train, X_test, y_train, y_test = train_test_split(df["text"].values, df["label"].values, test_size=0.2, random_state=42)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

            data_splits: dict ={}
            data_splits["training"] = Dataset.from_pandas(pd.DataFrame(columns=["text", "label"], data=list(zip(X_train, y_train))), preserve_index=False)
            data_splits["test"] = Dataset.from_pandas(pd.DataFrame(columns=["text", "label"], data=list(zip(X_test, y_test))), preserve_index=False)
            data_splits["validation"] = Dataset.from_pandas(pd.DataFrame(columns=["text", "label"], data=list(zip(X_valid, y_valid))), preserve_index=False)

            labelled_dataset: DatasetDict = DatasetDict(data_splits)

            labelled_dataset.push_to_hub(dataset_name, private=False)
            labelled_dataset = load_dataset(dataset_name, use_auth_token=True)
        return labelled_dataset