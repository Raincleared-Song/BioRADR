import json
from transformers import AutoTokenizer
from torch.optim import Adam, AdamW, SGD


class ConfigBase:
    """
        dataset   max_label   max_entity
        pretrain  24          24
        train     24          41
        dev       25          33
        test      42          40
    """
    # dmis-lab/biobert-base-cased-v1.1
    # allenai/scibert_scivocab_uncased
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    relation_num = 15  # contain NA
    bert_hidden = 768
    valid_instance_cnt = 14568
    optimizer_dict = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD
    }
    rank_result_path = 'CTDRED/rank_result'
    pair2triple_path = 'CTDRED/pair2triple.json'

    stat_path = 'CTDRED/stat.json'

    __file = open('CTDRED/relation_to_id.json')
    label2id = json.load(__file)
    __file.close()

    __file = open('CTDRED/id_to_relation.json')
    id2label = json.load(__file)
    __file.close()

    label2id: dict
    id2label: list

    data_path: dict
    score_path: dict
    title_path: dict
    batch_size: dict

    reader_num: int
    token_padding: int  # token reserved for each document
    entity_padding: dict  # entity reserved for each document
    mention_padding: int  # mention reserved for each entity
    train_sample_limit: int
    score_sample_limit: int
    test_sample_limit: int
    hidden_size: int

    do_validation: bool
    use_gpu: bool
    fp16: bool

    optimizer: str
    model_path: str
    model_name: str
    bert_path: str

    learning_rate: float
    weight_decay: int
    epoch_num: int

    output_step: int
    test_step: int
    lr_step_size: int
    lr_gamma: int
    train_steps: int  # optimizer step epoch number
    output_metric: str
    kept_pair_num: int  # number of pairs kept after predenoising

    positive_num: int
    negative_num: int
