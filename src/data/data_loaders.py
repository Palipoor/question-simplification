from src.utils import Simplifier
from datasets import load_dataset, Dataset
from numpy import load


def get_asset_data(split):
    if split == "train":
        asset_split = "validation"
    else:
        asset_split = "test"
    asset_data = load_dataset("src/data/asset.py", "simplification")[asset_split]
    instances = {"original": [], "simplification": []}
    for instance in asset_data:
        for simplification in instance["simplifications"]:
            instances["original"].append("simplify: " + instance["original"])
            instances["simplification"].append(simplification)
    dataset = Dataset.from_dict(instances)
    return dataset

def get_zest_data(split):
    """
    returns a dataset like:
    Dataset({
    features: ['inputs', 'targets'],
    num_rows: 2872
    })

    """
    if split != "train":
        split = "validation"
    zest_data = load_dataset("zest")[split]
    instances = {"inputs": [], "targets": []}
    for instance in zest_data:
        for answer in instance["answer"]: 
            instances["inputs"].append(
                f'zest question: {instance["question"]}?\n\n'
                f'zest context: {instance["context"]}\n\n')

            instances["targets"].append(answer)
    dataset = Dataset.from_dict(instances)
    return dataset


def get_simplified_zest(split, model_path,tokenizer_name):
    simplifier = Simplifier(model_path, tokenizer_name, length=128)
    if split != "train":
        split = "validation"
    zest_data = load_dataset("zest")[split]
    instances = {"inputs": [], "targets": []}
    for instance in zest_data:
        for answer in instance["answer"]: 
            question_simplified = simplifier.simplify(instance['question'])
            instances["inputs"].append(
                f'zest question: {question_simplified}?\n\n'
                f'zest context: {instance["context"]}\n\n')
        instances["targets"].append(answer)
    dataset = Dataset.from_dict(instances)
    return dataset


def get_turk_data(split):
    """
    returns a dataset like:
    Dataset({
    features: ['original', 'simplification'],
    num_rows: 2872
    })

    """
    if split == "train":
        turk_split = "validation"
    else:
        turk_split = "test"
    turk_data = load_dataset("turk")[turk_split]
    instances = {"original": [], "simplification": []}
    for instance in turk_data:
        for simplification in instance["simplifications"]:
            instances["original"].append("simplify: " +  instance["original"])
            instances["simplification"].append(simplification)
    dataset = Dataset.from_dict(instances)
    return dataset


def get_dataset(dataset_name, tokenizer, split, model_path = '', generate_tokenizer=''):
    def simplify_preprocess(data):
        inputs_encoded = tokenizer.encode_plus(
            text=data["original"],
            add_special_tokens=False,
            padding='max_length',
            return_attention_mask = True,
            max_length=128,
            truncation=True)
        input_ids = inputs_encoded['input_ids']
        input_attention_mask = inputs_encoded['attention_mask']
        decoder_ids = tokenizer.encode_plus(
            text=data['simplification'],
            add_special_tokens=False,
            padding='max_length',
            max_length=128,
            truncation=True)['input_ids']
        return {"input_ids": input_ids, "attention_mask" : input_attention_mask, "labels": decoder_ids}
    
    def zest_preprocess(data):
        inputs_encoded = tokenizer.encode_plus(
            text=data["inputs"],
            add_special_tokens=False,
            padding='max_length',
            return_attention_mask = True,
            max_length=128,
            truncation=True)
        input_ids = inputs_encoded['input_ids']
        input_attention_mask = inputs_encoded['attention_mask']
        decoder_ids = tokenizer.encode_plus(
            text=data['targets'],
            add_special_tokens=False,
            padding='max_length',
            max_length=128,
            truncation=True)['input_ids']
        return {"input_ids": input_ids, "attention_mask" : input_attention_mask, "labels": decoder_ids}


    if dataset_name == "turk":
        dataset = get_turk_data(split)
        return dataset.map(simplify_preprocess)
    elif dataset_name == "asset":
        dataset = get_asset_data(split)
        return dataset.map(simplify_preprocess)
    elif dataset_name == "zest":
        dataset = get_zest_data(split)
        return dataset.map(zest_preprocess)
    elif dataset_name == "zest-simplified":
        dataset = get_simplified_zest(split, model_path, generate_tokenizer)
        return dataset.map(zest_preprocess)