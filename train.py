import argparse

import click
import torch
import functools
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from src.data.data_loaders import *
from src.utils import *


@click.command()
@click.argument("dataset_name", metavar="<Dataset>")
@click.argument("model_name", metavar="<model_name>")
@click.option("--simplify_model_name", "-smn", type=str, default="")
@click.option("--simplify_tokenizer_name", "-stn", type=str, default="")
@click.option("--batch-size", "-bs", type=int, default=16)
@click.option("--checkpoint", "-chkpt", type = str, default = None)
@click.option("--run-name", "-run", type = str, default = None)
def main(dataset_name, model_name, simplify_model_name, 
        simplify_tokenizer_name, batch_size, checkpoint, run_name):
    if checkpoint:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    training_arguments = Seq2SeqTrainingArguments(
        output_dir=f"./checkpoints/{run_name}",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=16,
        logging_steps=1000,
        evaluation_strategy = "epoch",
        logging_first_step=True,
        save_total_limit=1,
        learning_rate=2e-5,
        predict_with_generate=True,
        report_to= ["tensorboard"]
    )

    train_dataset = None
    eval_dataset = None
    if dataset_name != "zest-simplified":
        train_dataset = get_dataset(dataset_name, tokenizer, "train")
        eval_dataset = get_dataset(dataset_name, tokenizer, "eval")
    else:
        train_dataset = get_dataset(dataset_name, tokenizer, "train", 
            model_path=simplify_model_name, generate_tokenizer=simplify_tokenizer_name)
        eval_dataset = get_dataset(dataset_name, tokenizer, "eval", 
            model_path=simplify_model_name, generate_tokenizer=simplify_tokenizer_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)

    if dataset_name in ["turk", "asset"]: 
        metr = functools.partial(compute_metrics_bleu,tokenizer=tokenizer)
    else:
        metr= None
    trainer = Seq2SeqTrainer(
        model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics= metr,
        tokenizer = tokenizer,
        data_collator=data_collator
    )
    trainer.train()


if __name__ == "__main__":
    main()
