import argparse

import click
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    training_args,
)
from data.data_loaders import *


@click.command()
@click.argument("mode", metavar="<Mode>")
@click.argument("dataset_name", metavar="<Dataset>")
@click.argument("model_name", metavar="<model_name>")
@click.option("--batch-size", "-bs", type=int, default=16)
def main(mode, dataset_name, model_name, batch_size):
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     help="train or evaluate mode",
    #     required=True,
    #     choices=["train", "evaluate"],
    # )
    # parser.add_argument("--dataset", type=str)
    # parser.add_argument("--model", type=str)
    # parser.add_argument("-load-ckpt", action="store_true")
    # parser.add_argument("--ckpt", type=str)
    # args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # if args.load_ckpt:
    #     pass  # TODO load from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    training_arguments = Seq2SeqTrainingArguments(
        # TODO update these. Just copied from another code I had lol
        output_dir="./output/",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=50,
        logging_steps=500,
        logging_first_step=True,
        save_steps=500,
        evaluation_strategy="steps",
        run_name="fooo",
        save_total_limit=2,
        learning_rate=1e-5,
    )
    train_dataset = get_dataset(dataset_name, tokenizer, "train")
    eval_dataset = get_dataset(dataset_name, tokenizer, "eval")
    print(eval_dataset)
    trainer = Seq2SeqTrainer(
        model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
