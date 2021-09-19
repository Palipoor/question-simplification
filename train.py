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
@click.argument("mode", metavar="<Mode>")
@click.argument("dataset_name", metavar="<Dataset>")
@click.argument("model_name", metavar="<model_name>")
@click.option("--batch-size", "-bs", type=int, default=16)
@click.option("--checkpoint", "-chkpt", type = str, default = None)
@click.option("--run-name", "-run", type = str, default = None)
def main(mode, dataset_name, model_name, batch_size, checkpoint, run_name):
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
        generation_max_length=70,
        generation_num_beams=5,
        do_predict= True,
        report_to= ["tensorboard"]
    )
    if checkpoint:
        training_arguments.resume_from_checkpoint = checkpoint
    train_dataset = get_dataset(dataset_name, tokenizer, "train")
    eval_dataset = get_dataset(dataset_name, tokenizer, "eval")
    data_collator = DataCollatorForSeq2Seq(tokenizer =tokenizer)
    trainer = Seq2SeqTrainer(
        model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=functools.partial(compute_metrics_bleu,tokenizer=tokenizer),
        tokenizer = tokenizer,
        data_collator=data_collator
    )
    trainer.train()


if __name__ == "__main__":
    main()
