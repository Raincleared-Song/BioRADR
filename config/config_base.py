import os
import json
from transformers import AutoTokenizer
from torch.optim import Adam, SGD
from transformers.optimization import AdamW


class ConfigBase:
    """
        dataset   max_label   max_entity
        pretrain  24          24
        train     24          41
        dev       25          33
        test      42          40
        cdr_train 14          21
        cdr_dev   16          20
        cdr_test  21          22
        cp_train  59          85     1716
        cp_dev    48          80     1584
        cp_test   54          113    2886
        cps_train 40          27     144
        cps_dev   32          27     162
        cps_test  30          22     99
    """
    tokenizer = AutoTokenizer.from_pretrained('../huggingface/scibert_scivocab_cased' if os.path.exists(
        '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased')
    # relation_num = 15  # CTDRED
    relation_num = 2  # cdr
    # relation_num = 24  # Chemprot
    bert_hidden = 768
    # valid_instance_cnt = 14568  # CTDRED
    # valid_instance_cnt = 997  # cdr
    # valid_instance_cnt = 8598  # CTD_binary
    # valid_instance_cnt = 7992  # CTD_binary2
    # valid_instance_cnt = 8872  # cdr_ctd
    # valid_instance_cnt = 13078   # ctd_cd_cg
    valid_instance_cnt = 20683  # ctd_all
    # valid_instance_cnt = 3466  # Chemprot_fr
    optimizer_dict = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD
    }
    # rank_result_path = 'CTDRED/rank_result'  # CTDRED
    rank_result_path = 'CTDRED/temp_range'  # CTDRED
    # rank_result_path = 'Chemprot/rank_result_fr'  # Chemprot sent
    pair2triple_path = 'CTDRED/pair2triple.json'  # CTDRED

    # __file = open('CTDRED/relation_to_id.json')
    # __file = open('CTDRED/relation_to_id_cdr.json')
    __file = open('CDR/rel2id.json')
    # __file = open('Chemprot/chemprot_relation_to_id.json')
    label2id = json.load(__file)
    __file.close()

    # __file = open('CTDRED/id_to_relation.json')
    __file = open('CDR/id2rel.json')
    # __file = open('Chemprot/chemprot_id_to_relation.json')
    id2label = json.load(__file)
    __file.close()

    seed = 66

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
    crop_documents: bool  # remove sentences not containing or surrounded by entities
    crop_mention_option: int  # remove redundant mentions, 0-not, 1-single, 2-more
    entity_marker_type: str

    do_validation: bool
    use_gpu: bool
    gpu_device: str
    fp16: bool

    optimizer: str
    model_path: str
    model_name: str
    bert_path: str

    learning_rate: float
    weight_decay: int
    adam_epsilon: float
    warmup_ratio: float
    from_epoch: int = -1
    epoch_num: int
    real_epoch_num: int

    output_step: int
    save_global_step: int
    test_step: int
    lr_step_size: int
    lr_gamma: int
    train_steps: int  # optimizer step epoch number
    output_metric: str
    kept_pair_num: int  # number of pairs kept after predenoising

    positive_num: int
    negative_num: int
