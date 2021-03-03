import json
from transformers import AutoTokenizer
from torch.optim import Adam, AdamW, SGD


class ConfigBase:
    # dmis-lab/biobert-base-cased-v1.1
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    relation_num = 97  # contain NA
    bert_hidden = 768
    valid_instance_cnt = 12323
    optimizer_dict = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD
    }
    rank_result_path = 'DocRED/rank_result'
    pair2triple_path = 'DocRED/pair2triple.json'

    stat_path = 'CTDRED/stat.json'

    label2id: dict
    id2label: list

    data_path: dict
    score_path: dict
    title_path: dict
    batch_size: dict

    reader_num: int
    token_padding: int  # token reserved for each document
    entity_padding: int  # entity reserved for each document
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

    __file = open('DocRED/label2id.json')
    label2id = json.load(__file)
    __file.close()

    __file = open('DocRED/id2label.json')
    id2label = json.load(__file)
    __file.close()

    positive_num: int
    negative_num: int
