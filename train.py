import argparse
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, training_args
from .data_loaders import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='train or evaluate mode', required=True, choices=['train', 'evaluate'])
    parser.add_argument('--dataset', type = str)
    args = parser.parse_args()
    MODEL_NAME = 't5-large'

    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if args.mode == 'train':
        training_arguments = TrainingArguments(  #TODO update these. Just copied from another code I had lol
                do_train=True,
                do_eval=True,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=50,
                logging_steps=500,
                logging_first_step=True,
                save_steps=500,
                evaluation_strategy="steps", 
                run_name  = "fooo",
                save_total_limit = 2,
                learning_rate = 1e-5
        )
        train_dataset = get_dataset(args.dataset, 'train')
        eval_dataset = get_dataset(args.dataset, 'eval')
        trainer = Trainer(args=training_arguments, train_dataset = train_dataset, eval_dataset=eval_dataset)
        trainer.train()
    else:
        pass
