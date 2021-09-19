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
            instances["original"].append(instance["original"])
            instances["simplification"].append(simplification)
    dataset = Dataset.from_dict(instances)
    return dataset

def get_zest_data(split):
    dataset = load_dataset('zest',split)
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
            instances["original"].append(instance["original"])
            instances["simplification"].append(simplification)
    dataset = Dataset.from_dict(instances)
    return dataset


def get_dataset(dataset_name, tokenizer, split):
    def preprocess(data):
        # add "simplify" as task-specific prefix
        data["original"] = "simplify: " + data['original'] 
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

    if dataset_name == "turk":
        dataset = get_turk_data(split)
        return dataset.map(preprocess)
    elif dataset_name == "asset":
        dataset = get_asset_data(split)
        return dataset.map(preprocess)
