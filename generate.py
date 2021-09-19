from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer)
from tqdm import tqdm
import click

@click.command()
@click.argument("checkpoint", metavar="<model_name>")
@click.argument("tokenizer", metavar="<tokenizer>")
@click.argument("source_dataset", metavar="<source_dataset>")
@click.option("--max_length", "-max_len" , type = int, default=128)
@click.option("--beam", "-b" , type = int, default=5)
def main(checkpoint,tokenizer, source_dataset, max_length, beam):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
