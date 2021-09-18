from datasets import load_dataset, Dataset
from numpy import load

def get_asset_data():
    pass

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
    pass