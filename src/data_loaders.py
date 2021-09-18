from datasets import load_dataset, Dataset
from numpy import load

def get_asset_data(split):
    asset = load_dataset("src/data/asset.py",'simplification')
    return asset
def get_ambiqa_data():
    pass

def get_zest_data():
    pass

def get_turk_data(split):   
    """
    returns a dataset like:
    Dataset({
    features: ['original', 'simplification'],
    num_rows: 2872
    })
    
    """
    if split == 'train':
        turk_split = 'evaluation'
    else:
        turk_split = 'test'
    turk_data = load_dataset('turk')[turk_split]
    instances = {
        'original': [],
        'simplification': []
    }
    for instance in turk_data:
        for simplification in instance['simplifications']:
            instances['original'].append(instance['original'])
            instances['simplification'].append(simplification)
    dataset = Dataset.from_dict(instances)
    return dataset
    
def get_dataset(dataset_name,tokenizer, split):
    def preprocess(data) : 
        input_ids = tokenizer(text=data['original'], return_tensors='pt').input_ids
        decoder_ids = tokenizer(text=data['simplification'], return_tensors = 'pt').input_ids
        return {'input_ids': input_ids, 'decoder_input_ids': decoder_ids}

    if dataset_name == 'turk': 
        dataset = get_turk_data(split)
        return dataset.map(preprocess)
