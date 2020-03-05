import os

import json
import argparse

from problems import CONFIG, TranslateManyToMany
from datasets import add_multilingual_files, add_mixed_files, create_data_config

from tensor2tensor.problems import problem

from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor.utils import registry
from tensor2tensor import models
from tensor2tensor import problems

from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment


def save_data_config(training_dir, testing_dir):
    training_files, testing_files = [], []

    training_files, langs1 = add_mixed_files(
        os.path.join(training_dir, "train.a"), 
        os.path.join(training_dir, "train.b"), 
        os.path.join(training_dir, "train.langpairs"), 
        training_files)
    
    testing_files, langs2 = add_multilingual_files(testing_dir, "test", testing_files)

    languages = sorted(list(set(langs1) | set(langs2)))
    config = create_data_config(training_files, testing_files, languages)
    with open("data_config.json", "w") as fp:
        json.dump(config, fp)
    
    return config

def preprocessing(args):
    config = save_data_config(args.training_dir, args.testing_dir)
    CONFIG.update(config)
    t2t_problem = problem(TranslateManyToMany.name)
    t2t_problem.generate_data(args.data_dir, args.temp_dir)

def _init_network(args):
    with open("data_config.json") as fp:
        config = json.load(fp)
    CONFIG.update(config)

    MODEL = "transformer"
    HPARAMS = "transformer_tpu"
    train_steps = 1000000
    eval_steps = 10
    save_checkpoints_steps = 10000
    schedule = "continuous_train_and_eval"

    hparams = create_hparams(HPARAMS)
    print(json.loads(hparams.to_json()))

    # fix TPU zone
    from tensorflow.distribute.cluster_resolver import TPUClusterResolver
    TPUClusterResolver.__init__.__defaults__ = (args.tpu_name, args.tpu_zone, None, 'worker', None, None, 'default', None, None)
    print(TPUClusterResolver.__init__.__defaults__)

    RUN_CONFIG = create_run_config(
        model_dir=args.model_dir,
        model_name=MODEL,
        save_checkpoints_steps= save_checkpoints_steps,
        use_tpu=True,
        cloud_tpu_name=args.tpu_name,
    )
    print(type(RUN_CONFIG))
    tensorflow_exp_fn = create_experiment(
            run_config=RUN_CONFIG,
            hparams=hparams,
            model_name=MODEL,
            problem_name=TranslateManyToMany.name,
            data_dir=args.data_dir,
            train_steps=train_steps,
            eval_steps=eval_steps,
            use_tpu=True,
            schedule=schedule,
            #use_xla=True # For acceleration
        )
    return tensorflow_exp_fn

def training(args):
    tensorflow_exp_fn = _init_network(args)
    out = tensorflow_exp_fn.train(max_steps=300000)
    print(out)

def do_eval(args):
    tensorflow_exp_fn = _init_network(args)
    out = tensorflow_exp_fn.evaluate()
    print(out)

def main(args):
    if args.action == 'prepro':
        preprocessing(args)
    if args.action == 'train':
        training(args)
    if args.action == 'eval':
        do_eval(args)

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("action", choices=['prepro', 'train', 'eval'])
    argp.add_argument("--data-dir", type=str, required=True)
    argp.add_argument("--model-dir", type=str, required=True)
    argp.add_argument("--temp-dir", type=str, default="/tmp")
    argp.add_argument("--training-dir", type=str, required=False)
    argp.add_argument("--testing-dir", type=str, required=False)

    argp.add_argument("--tpu-name", type=str, default="tpu-1")
    argp.add_argument("--tpu-zone", type=str, default="europe-west4-a")

    args = argp.parse_args()
    import tensorflow as tf
    from functools import partial
    main(args)
