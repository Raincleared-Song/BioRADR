import json
from .config_base import ConfigBase


class ConfigFineTune(ConfigBase):
    data_path = {
        'train': 'CTDRED/train_mixed.json',
        'valid': 'CTDRED/dev.json',
        'test': 'CTDRED/test.json'
    }
    # data_path = {
    #     'train': 'CTDRED/cdr_train.json',
    #     'valid': 'CTDRED/cdr_dev.json',
    #     'test': 'CTDRED/cdr_test.json'
    # }
    # data_path = {
    #     'train': 'Chemprot/chemprot_train_sent_fr.json',
    #     'valid': 'Chemprot/chemprot_dev_sent_fr.json',
    #     'test': 'Chemprot/chemprot_test_sent_fr.json'
    # }
    batch_size = {
        'train': 4,
        'valid': 16,
        'test': 16
    }
    # score_path = None
    score_path = {
        'train': f'{ConfigBase.rank_result_path}/train_mixed_score.npy',
        'valid': f'{ConfigBase.rank_result_path}/dev_score.npy',
        'test': f'{ConfigBase.rank_result_path}/test_score.npy'
    }
    # score_path = {
    #     'train': f'{ConfigBase.rank_result_path}/chemprot_train_sent_fr_score.npy',
    #     'valid': f'{ConfigBase.rank_result_path}/chemprot_dev_sent_fr_score.npy',
    #     'test': f'{ConfigBase.rank_result_path}/chemprot_test_sent_fr_score.npy'
    # }
    title_path = {
        'train': f'{ConfigBase.rank_result_path}/train_mixed_title.json',
        'valid': f'{ConfigBase.rank_result_path}/dev_title.json',
        'test': f'{ConfigBase.rank_result_path}/test_title.json'
    }
    # title_path = {
    #     'train': f'{ConfigBase.rank_result_path}/chemprot_train_sent_fr_title.json',
    #     'valid': f'{ConfigBase.rank_result_path}/chemprot_dev_sent_fr_title.json',
    #     'test': f'{ConfigBase.rank_result_path}/chemprot_test_sent_fr_title.json'
    # }

    reader_num = 30
    bert_path = 'dmis-lab/biobert-base-cased-v1.1'  # can be modified by terminal args
    token_padding = 512  # token reserved for each document
    # token_padding = 448  # token reserved for each document, cps
    entity_padding = None  # entity reserved for each document
    mention_padding = 3  # mention reserved for each entity
    # mention_padding = 1
    train_sample_limit = 32  # max positive label number is 24
    # train_sample_limit = 14  # cdr
    test_sample_limit = 1600
    # test_sample_limit = 480  # cdr
    # train_sample_limit = 80
    # test_sample_limit = 2886  # #chemical * #gene
    # train_sample_limit = 40
    # test_sample_limit = 162  # #chemical * #gene
    score_sample_limit = 50
    kept_pair_num = 50
    # score_sample_limit = 80
    # kept_pair_num = 80
    # score_sample_limit = 40
    # kept_pair_num = 40
    do_validation = True
    use_gpu = True
    gpu_device = 'cuda:0'
    hidden_size = 256

    optimizer = 'adamw'
    learning_rate = 2e-5
    weight_decay = 0
    epoch_num = 60

    output_step = 1
    test_step = 1
    model_path = 'checkpoint'
    model_name = 'ctd_finetune_lr02_t07'
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

    score_threshold = 0.7
    test_score_threshold = 0.99999

    use_extra = False
    if use_extra:
        __file__ = open('config/auto.json', encoding='utf-8')
        __extra__ = json.load(__file__)
        if 'model_name' in __extra__:
            model_name = __extra__['model_name']
        else:
            raise RuntimeError('model_name')
        if 'learning_rate' in __extra__:
            learning_rate = __extra__['learning_rate']
        else:
            raise RuntimeError('learning_rate')
        if 'score_threshold' in __extra__:
            score_threshold = __extra__['score_threshold']
        else:
            score_threshold = None
        if 'test_score_threshold' in __extra__:
            test_score_threshold = __extra__['test_score_threshold']
        else:
            test_score_threshold = None
        if 'score_path' in __extra__:
            score_path = __extra__['score_path']
        else:
            score_path = None
        __file__.close()

    assert not ((score_threshold is None) ^ (test_score_threshold is None))
