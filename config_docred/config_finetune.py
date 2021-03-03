from .config_base import ConfigBase


class ConfigFineTune(ConfigBase):
    data_path = {
        'train': 'DocRED/train_annotated.json',
        'valid': 'DocRED/dev.json',
        'test': 'DocRED/test.json'
    }
    batch_size = {
        'train': 4,
        'valid': 4,
        'test': 4
    }
    score_path = {
        'train': f'{ConfigBase.rank_result_path}/train_annotated_score.npy',
        'valid': f'{ConfigBase.rank_result_path}/dev_score.npy',
        'test': f'{ConfigBase.rank_result_path}/test_score.npy'
    }
    title_path = {
        'train': f'{ConfigBase.rank_result_path}/train_annotated_title.json',
        'valid': f'{ConfigBase.rank_result_path}/dev_title.json',
        'test': f'{ConfigBase.rank_result_path}/test_title.json'
    }

    reader_num = 30
    bert_path = 'bert-base-uncased'  # can be modified by terminal args
    token_padding = 512  # token reserved for each document
    entity_padding = None  # entity reserved for each document
    mention_padding = 3  # mention reserved for each entity
    train_sample_limit = 90
    test_sample_limit = 1800
    score_sample_limit = 60
    do_validation = True
    use_gpu = True
    hidden_size = 256

    optimizer = 'adamw'
    learning_rate = 1e-5
    weight_decay = 0
    epoch_num = 45

    output_step = 1
    test_step = 1
    model_path = 'checkpoint'
    model_name = 'finetune'
    fp16 = False
    lr_step_size = 1
    lr_gamma = 1
    train_steps = 1
    output_metric = 'f1_auc_metric'
    kept_pair_num = 60

    positive_num = None
    negative_num = None

    use_entity_type = False
    type_embed_size = 128
    use_loss_weight = False
    loss_weight: list
