import numpy as np
from transformers import EvalPrediction
from datasets import load_metric
from typing import Dict

from transformers import AutoModelForSeq2SeqLM,AutoTokenizer


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics_bleu(prediction, tokenizer):
    metric = load_metric("sacrebleu")
    preds, labels = prediction
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


class Simplifier:
    def __init__(self, path, tokenizer, beam = 5, length = 128):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.beam = beam
        self.length = length
    def simplify(self,text):
        inputs_encoded = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=False,
            padding='max_length',
            return_attention_mask = True,
            max_length=128,
            truncation=True, return_tensors='pt')
        input_ids = inputs_encoded['input_ids']
        output = self.model.generate(input_ids = input_ids, num_beams=self.beam, max_length = self.length, do_sample = True, early_stopping=True, repetition_penalty= 2.0)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]