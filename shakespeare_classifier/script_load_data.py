
import os
from dotenv import load_dotenv, find_dotenv
from dataloader import ShakespeareanDataset
import argparse


def create_or_load_dataset(force_reload:bool= False):
    _ = load_dotenv(find_dotenv())
    os.getenv('HUGGINGFACE_TOKEN')
    return ShakespeareanDataset().get_dataset(force_reload=force_reload)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--force_reload", type=bool, default=False)
    args = parser.parse_args()
    print(args.force_reload, type(args.force_reload))
    ds = create_or_load_dataset(args.force_reload)
    print(ds)
    
