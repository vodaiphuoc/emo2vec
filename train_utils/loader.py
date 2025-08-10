from torch.utils.data import DataLoader
import datasets
from typing import List, Dict, Union, Tuple
from .training_cfg import TrainingConfig
import torch
import random

ARR_DTYPE = torch.float32
LABEL_DTYPE = torch.int64

MAX_ARR_LENGTH = 90000

EMOTION2IDS = {"happy": 0, "neutral": 1, "sad": 2, "angry": 3}

def _collator(examples: List[Dict[str, Union[int, str, List[float]]]]):
    # batch_max_length = max([len(exp['data_array']) for exp in examples])
    batch_max_length =  MAX_ARR_LENGTH

    source_tensor_list = []
    padding_mask = torch.BoolTensor(torch.Size([len(examples), batch_max_length])).fill_(False)
    labels = []

    for _ith, example in enumerate(examples):
        current_length = len(example['data_array'])
        if current_length > MAX_ARR_LENGTH:
            example['data_array'] = example['data_array'][:MAX_ARR_LENGTH]
        else:
            example['data_array'].extend([0]*(batch_max_length - current_length))
            padding_mask[_ith, current_length: ] = True
        
        source_tensor_list.append(torch.tensor(example['data_array'], dtype= ARR_DTYPE))
        labels.append(example['emotion_id'])

    stacked_source = torch.stack(source_tensor_list, dim=0)
    
    return (
        {
            "source": stacked_source,
            "padding_mask": padding_mask
        },
        torch.tensor(labels, dtype= LABEL_DTYPE)
    )


def _pre_process_dataset(dataset: datasets.Dataset)->datasets.Dataset:
    r"""
    Select only main columns
    """
    return dataset.map(
        lambda example: {
            "data_array": example['path']['array'],
            'emotion_id': int(example['emotion_id']),
            'emotion': example['emotion']
        }, 
        batched = False,
        remove_columns = dataset.column_names
    )

def train_test_split(ds: datasets.Dataset, ratio: float)->Tuple[datasets.Dataset]:
    emotion_set = set(ds['emotion'])

    emotion2ids_mapping = {emo: [] for emo in emotion_set}

    for _ith, exp in enumerate(ds):
        emotion2ids_mapping[exp['emotion']].append(_ith)

    train_ds_list = []
    test_ds_list = []
    from copy import deepcopy
    for k, v in emotion2ids_mapping.items():
        ids = deepcopy(v)
        random.shuffle(ids)
        test_length = int(len(ids)*ratio)
        train = ids[:(len(ids)-test_length)]
        test = ids[(len(ids)-test_length):]

        train_ds_result = ds.select(indices= train, keep_in_memory=True)
        test_ds_result = ds.select(indices= test, keep_in_memory=True)

        train_ds_list.append(train_ds_result)
        test_ds_list.append(test_ds_result)

    train_concat = datasets.concatenate_datasets(train_ds_list)
    test_concat = datasets.concatenate_datasets(test_ds_list)
    
    train_concat = train_concat.shuffle()
    test_concat = test_concat.shuffle()

    return (
        _pre_process_dataset(train_concat),
        _pre_process_dataset(test_concat)
    )

def get_dataloader(training_config: TrainingConfig):

    # source dataset doesnt have train and test sets
    dataset = datasets.load_dataset("hustep-lab/ViSEC")['train']
    train_ds, test_ds = train_test_split(
        ds= dataset, 
        ratio= training_config.test_size
    )

    print('check label in train_ds: ', set(train_ds['emotion']), train_ds.column_names)
    for emo in set(train_ds['emotion']):
        print(emo, len([ids for ids, exp in enumerate(train_ds) if exp['emotion'] == emo]))

    print('check label in test_ds: ', set(test_ds['emotion']), test_ds.column_names)
    for emo in set(test_ds['emotion']):
        print(emo, len([ids for ids, exp in enumerate(test_ds) if exp['emotion'] == emo]))

    train_dataloader = DataLoader(
        train_ds, 
        batch_size= training_config.batch_size,
        num_workers= training_config.num_workers,
        pin_memory= True,
        shuffle=True,
        prefetch_factor= training_config.prefetch_factor,
        collate_fn=_collator
    )

    test_dataloader = DataLoader(
        test_ds, 
        batch_size= training_config.batch_size,
        num_workers= training_config.num_workers,
        pin_memory= True,
        shuffle=False,
        prefetch_factor= training_config.prefetch_factor,
        collate_fn=_collator
    )

    return train_dataloader, test_dataloader