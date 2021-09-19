import click
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer)
from src.data.data_loaders import *
from src.utils import *
import torch
from tqdm import tqdm

@click.command()
@click.argument("model_name", metavar="<model>")
@click.argument("checkpoint", metavar="<model>")
@click.option("--beam", "-b", type=int, default=5)
@click.option("--length", "-l", type=int, default=128)
@click.option('--simplified', type = str, default=False)
@click.option("--simplifier_model", "-simplifier", type = str, default = None)
@click.option("--simplifier_tokenizer", "-simplifier_tokenizer", type = str, default = None)
def main(model_name, checkpoint, beam, length, simplified,simplifier_model, simplifier_tokenizer):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if simplified == 'yes':
        data = get_dataset("zest-simplified", tokenizer, 'validation')
    else:
        data = get_dataset("zest", tokenizer, 'validation')
    counter = 0
    predictions = []
    for d in tqdm(data):
        generated = model.generate(input_ids = torch.tensor([d['input_ids']]), num_beams=beam, max_length = length, do_sample = True, early_stopping=True, repetition_penalty= 2.0)
        decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        predictions.append(decoded_preds)
    with open('results.txt', 'w') as f:
        for p in predictions:
            f.writeline(p)
if __name__ == '__main__':
    main()