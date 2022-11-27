import os
from transformers import AutoTokenizer
from torch.optim import Adam, SGD
from transformers.optimization import AdamW


class ConfigBase:
    tokenizer = AutoTokenizer.from_pretrained('../huggingface/scibert_scivocab_cased' if os.path.exists(
        '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased')
    relation_num = 2
    bert_hidden = 768
    valid_instance_cnt = 20683  # ctd_all
    optimizer_dict = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD
    }
    rank_result_path = 'CTDRED/temp_range'

    label2id = {"NA": 0, "Pos": 1}
    id2label = ["NA", "Pos"]

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
    model_class: str
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


class ConfigFineTune(ConfigBase):
    data_path = {
        'train': 'CTDRED/ctd_train.json',
        'valid': 'CTDRED/ctd_dev.json',
        'test': 'CTDRED/ctd_test.json'
    }
    batch_size = {
        'train': 4,
        'valid': 16,
        'test': 16
    }

    reader_num = 4
    bert_path = '../huggingface/scibert_scivocab_cased' if os.path.exists(
        '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased'
    token_padding = 1024  # token reserved for each document
    entity_padding = None  # entity reserved for each document
    mention_padding = 3  # mention reserved for each entity
    train_sample_limit = 270  # CTD_binary
    train_sample_number = 60  # cdr_cdr_neg_sample
    test_sample_limit = 294  # CTD_binary
    score_sample_limit = 50
    kept_pair_num = 50
    do_validation = True
    use_gpu = True
    gpu_device = 'cuda:0'
    hidden_size = 256

    optimizer = 'adamw'
    learning_rate = 1e-5  ### ATTENTION! 2e-5 for cdr, 4e-4 group default for DocuNet
    weight_decay = 0
    adam_epsilon = 1e-6
    from_epoch = 0
    epoch_num = 30
    real_epoch_num = 1
    warmup_ratio = 0.06

    output_step = 1
    save_global_step = -1
    crop_documents = False
    crop_mention_option = 0
    entity_marker_type = 't'
    assert crop_mention_option in [0, 1, 2, 3, 4]
    assert entity_marker_type in ['mt', 'm', 't', 't-m', 'm*']
    test_step = 1
    model_path = 'checkpoint'
    model_name = 'ctd_pretrain_pre_sci_type'
    model_class = 'FineTuneModel'
    fp16 = False
    lr_step_size = 1
    lr_gamma = 1
    train_steps = 1
    output_metric = 'f1_auc_metric'

    positive_num = None
    negative_num = None

    na_weight = 1  # loss weight of NA
    use_entity_type = False
    type_embed_size = 128
    use_loss_weight = False
    loss_weight = [1] * ConfigBase.relation_num
    use_stat = 'train_mixed'

    score_threshold = None
    test_score_threshold = None

    do_negative_sample = False  # if do negative samples

    use_group_bilinear = True
    use_logsumexp = False
    output_score_type = 'softmax'
    bilinear_block_size = 64
    only_chem_disease = True

    assert output_score_type in ['pos', 'sigmoid', 'softmax', 'diff', 'sig_diff']
    assert not ((score_threshold is None) ^ (test_score_threshold is None))
