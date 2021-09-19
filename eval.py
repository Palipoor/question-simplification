import click
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer)
from src.data.data_loaders import *
from src.utils import *
import torch

@click.command()
@click.argument("dataset_name", metavar="<Dataset>")
@click.argument("model", metavar="<model>")
@click.argument("checkpoint", metavar="<model>")
@click.option("--beam", "-b", type=int, default=5)
@click.option("--length", "-l", type=int, default=128)
@click.option("--simplifier_model", "-simplifier", type = str, default = None)
@click.option("--simplifier_tokenizer", "-simplifier_tokenizer", type = str, default = None)
def main(dataset_name, model, checkpoint, beam, length, simplifier_model, simplifier_tokenizer):
    bleu = load_metric("sacrebleu")
    sari = load_metric("sari")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if 'simplified' in dataset_name:
        data = get_dataset(dataset_name, tokenizer, 'validation',simplifier_model,simplifier_tokenizer)
    else:    
        data = get_dataset(dataset_name, tokenizer, 'validation')
    total_bleu = 0
    total_sari=0
    for d in data:
        references = np.where(d['labels'] != -100, d['labels'], tokenizer.pad_token_id)
        decoded_references = [tokenizer.batch_decode([references], skip_special_tokens=True)]
        generated = model.generate(input_ids = torch.tensor([d['input_ids']]), num_beams=beam, max_length = length, do_sample = True, early_stopping=True, repetition_penalty= 2.0)
        decoded_preds = [tokenizer.batch_decode(generated, skip_special_tokens=True)[0]]
        result = bleu.compute(predictions=decoded_preds, references=decoded_references)
        total_bleu += result['score']
        source = d['input_ids']
        if 'simplified' in dataset_name:
            decoded_source = [tokenizer.batch_decode(source, skip_special_tokens=True)[0]]
            result = sari.compute(sources = decoded_source, predictions=decoded_preds, references=decoded_references)
            total_sari += result['sari']
    total_bleu = total_bleu / len(data)
    total_sari = total_sari / len(data)
    print('bleu ', total_bleu)
    if 'simplified' in dataset_name:
        print('sari ', total_sari)

    
if __name__ == '__main__':
    main()