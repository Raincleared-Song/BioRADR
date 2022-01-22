import os
import json
from .config_base import ConfigBase


class ConfigFineTune(ConfigBase):
    data_path = {
        'train': '../project-1/CDR/train_cdr.json',
        'valid': '../project-1/CDR/dev_cdr.json',
        'test': '../project-1/CDR/test_cdr.json'
    }
    # data_path = {
    #     'train': 'CTDRED/train_mixed.json',
    #     'valid': 'CTDRED/dev.json',
    #     'test': 'CTDRED/test.json'
    # }
    # data_path = {
    #     'train': '../project-1/CDR/train_cdr_ctd_pos.json',
    #     'valid': '../project-1/CDR/dev_cdr_ctd_pos.json',
    #     'test': '../project-1/CDR/test_cdr_ctd_pos.json',
    #     'negative_train': '../project-1/CTDRED/negative_train_mixed_binary_pos.json',
    #     'negative_valid': '../project-1/CTDRED/negative_dev_binary_pos.json',
    #     'negative_test': '../project-1/CTDRED/negative_test_binary_pos.json'
    # }
    # data_path = {
    #     'train': 'CDR/train_cdr.json',
    #     'valid': 'CDR/dev_cdr.json',
    #     'test': 'CDR/test_cdr.json'
    # }
    batch_size = {
        'train': 4,
        'valid': 16,
        'test': 16
    }
    # score_path = {
    #     'train': f'{ConfigBase.rank_result_path}/train_mixed_score.npy',
    #     'valid': f'{ConfigBase.rank_result_path}/dev_score.npy',
    #     'test': f'{ConfigBase.rank_result_path}/test_score.npy'
    # }
    score_path = {
        'train': f'{ConfigBase.rank_result_path}/train_mixed_binary_pos_score.npy',
        'valid': f'{ConfigBase.rank_result_path}/dev_binary_pos_score_17.npy',
        'test': f'{ConfigBase.rank_result_path}/test_binary_pos_score.npy',
        'negative_train': f'{ConfigBase.rank_result_path}/negative_train_mixed_binary_pos_score.npy',
        'negative_valid': f'{ConfigBase.rank_result_path}/negative_dev_binary_pos_score.npy',
        'negative_test': f'{ConfigBase.rank_result_path}/negative_test_binary_pos_score.npy'
    }
    # title_path = {
    #     'train': f'{ConfigBase.rank_result_path}/train_mixed_title.json',
    #     'valid': f'{ConfigBase.rank_result_path}/dev_title.json',
    #     'test': f'{ConfigBase.rank_result_path}/test_title.json'
    # }
    title_path = {
        'train': f'{ConfigBase.rank_result_path}/train_mixed_binary_pos_pmid2range.json',
        'valid': f'{ConfigBase.rank_result_path}/dev_binary_pos_pmid2range_17.json',
        'test': f'{ConfigBase.rank_result_path}/test_binary_pos_pmid2range.json',
        'negative_train': f'{ConfigBase.rank_result_path}/negative_train_mixed_binary_pos_pmid2range.json',
        'negative_valid': f'{ConfigBase.rank_result_path}/negative_dev_binary_pos_pmid2range.json',
        'negative_test': f'{ConfigBase.rank_result_path}/negative_test_binary_pos_pmid2range.json'
    }

    reader_num = 8
    bert_path = '../huggingface/scibert_scivocab_cased' if os.path.exists(
        '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased'
    token_padding = 1024  # token reserved for each document
    entity_padding = None  # entity reserved for each document
    mention_padding = 3  # mention reserved for each entity
    # mention_padding = 1
    # train_sample_limit = 32  # max positive label number is 24
    train_sample_limit = 104  # cdr
    # train_sample_limit = 270  # CTD_binary
    train_sample_number = 60  # cdr_cdr_neg_sample
    # test_sample_limit = 1600
    test_sample_limit = 117  # cdr
    # test_sample_limit = 294  # CTD_binary
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
    adam_epsilon = 1e-6
    epoch_num = 40
    warmup_ratio = 0.06

    output_step = 1
    save_global_step = -1
    crop_documents = False
    crop_mention_option = 0
    entity_marker_type = 't'
    assert crop_mention_option in [0, 1, 2]
    assert entity_marker_type in ['mt', 'm', 't']
    test_step = 1
    model_path = 'checkpoint'
    model_name = 'cdr_finetune_sci_type'
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
