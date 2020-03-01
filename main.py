import json
import argparse

from .problems import CONFIG, TranslateManyToMany
from .datasets import add_multilingual_files, add_prefixed_files, create_data_config

from tensor2tensor import problems


def save_data_config(training_dir, testing_dir):
    training_files, testing_files = [], []

    training_files, langs1 = add_prefixed_files(
        os.path.join(training_dir, "raw.src.txt"), 
        os.path.join(training_dir, "raw.tgt.txt"), 
        training_files)
    
    testing_files, langs2 = add_multilingual_files(testing_dir, "data", testing_files)

    languages = sorted(list(set(langs1) | set(langs2)))
    config = create_data_config(training_files, testing_files, languages)
    with open("data_config.json", "w") as fp:
        json.dump(config, fp)
    
    return config

def preprocessing(args):
    config = save_data_config(args.training_dir, args.testing_dir)
    CONFIG.update(config)
    t2t_problem = problems.problem(TranslateManyToMany.name)
    t2t_problem.generate_data(args.data_dir, args.temp_dir)

def main(args):
    if args.action == 'prepro':
        preprocessing(args)

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("action", choices=['prepro', 'train'])
    argp.add_argument("--data-dir", type=str, required=True)
    argp.add_argument("--temp-dir", type=str, default="/tmp")
    args = argp.parse_args()
    main(args)
