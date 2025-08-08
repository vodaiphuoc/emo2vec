from torch.utils.data import DataLoader
import datasets
from typing import List, Dict, Union
from .training_cfg import TrainingConfig
import torch

ARR_DTYPE = torch.float16
LABEL_DTYPE = torch.int64

MAX_ARR_LENGTH = 40000

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
    print('_collator, stacked_source shape: ', stacked_source.shape)
    return (
        {
            "source": stacked_source,
            # "padding_mask": padding_mask
        },
        torch.tensor(labels, dtype= LABEL_DTYPE)
    )

def get_dataloader(training_config: TrainingConfig):

    # source dataset doesnt have train and test sets
    dataset = datasets.load_dataset("hustep-lab/ViSEC")['train']

    dataset = dataset.train_test_split(test_size=training_config.test_size, shuffle=True)

    train_ds = _pre_process_dataset(dataset['train'])
    test_ds = _pre_process_dataset(dataset['test'])

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