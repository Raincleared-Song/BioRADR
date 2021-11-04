import json
import subprocess


if __name__ == '__main__':
    use_cp = False
    if use_cp:
        out_file = open('test_cps_log.txt', 'a', encoding='utf-8')
        out_train = open('test_cps_train_log.txt', 'a', encoding='utf-8')
        rank_result_path = 'CTDRED/rank_result_fr'
    else:
        out_file = open('test_log.txt', 'a', encoding='utf-8')
        out_train = open('test_train_log.txt', 'a', encoding='utf-8')
        rank_result_path = 'CTDRED/rank_result'
    test_list = [
        # ('test_ctd_finetune1_t00_lr01', 44, 1e-5, 0, 0),
        # ('test_ctd_finetune1_t01_lr01', 31, 1e-5, 1, 1),
        # ('test_ctd_finetune1_t02_lr01', 44, 1e-5, 2, 2),
        # ('test_ctd_finetune1_t02_lr02', 14, 2e-5, 2, 2),
        # ('test_ctd_finetune1_t02_lr05', 4, 5e-5, 2, 2),
        # ('test_ctd_finetune1_t02_lr10', 2, 1e-4, 2, 2),
        # ('test_ctd_finetune1_t_2_lr02', 25, 2e-5, -2, -2),
        # ('test_ctd_finetune1_t_1_lr02', 42, 2e-5, -1, -1),
        # ('test_ctd_finetune1_t00_lr02', 27, 2e-5, 0, 0),
        # ('test_ctd_finetune1_t00_lr03', 9, 3e-5, 0, 0),
        # ('test_ctd_finetune1_t00_lr04', 17, 4e-5, 0, 0),
        # ('test_ctd_finetune1_tn_lr02', 39, 2e-5, None, None),
        # ('test_ctd_finetune1_tn_lr02_a', 51, 2e-5, None, None),
        # ('test_ctd_finetune1_tn_lr02_m1', 54, 2e-5, None, None),
        # ('test_ctd_finetune1_tn_lr02_m2', 37, 2e-5, None, None),
        # ('test_ctd_finetune1_tn_lr02_m3', 41, 2e-5, None, None)
        # ('test_ctd_finetune1_t0100_lr02', 25, 2e-5, 1, 0),
        # ('test_ctd_finetune1_t0200_lr02', 15, 2e-5, 2, 0),
        # ('test_ctd_finetune1_t_100_lr02', 14, 2e-5, -1, 0),
        # ('test_ctd_finetune1_t_200_lr02', 14, 2e-5, -2, 0),
        # ('ctd_finetune_lr02', 33, 2e-5, None, None),
        # ('test_ctd_finetune_t0405_lr02', 6, 2e-5, 0.4, 0.5),
        # ('test_ctd_finetune_t0505_lr02', 25, 2e-5, 0.5, 0.5),
        # ('test_ctd_finetune_t0605_lr02', 19, 2e-5, 0.6, 0.5),
        # ('test_ctd_finetune_t0705_lr02', 10, 2e-5, 0.7, 0.5),
        # ('test_ctd_finetune_t0805_lr02', 35, 2e-5, 0.8, 0.5),
        # ('test_ctd_finetune_t0905_lr02', 13, 2e-5, 0.9, 0.5),
        # ('test_ctd_finetune_t09905_lr02', 14, 2e-5, 0.99, 0.5),
        # ('test_ctd_finetune_t099905_lr02', 8, 2e-5, 0.999, 0.5),
        # ('test_ctd_finetune_t0999905_lr02', 19, 2e-5, 0.9999, 0.5),
        # ('test_ctd_finetune_t09999905_lr02', 1, 2e-5, 0.99999, 0.5),
        # ('test_cps_finetune_t0505_lr01', 28, 1e-5, 0.5, 0.5),
        # ('test_cps_finetune_t0505_lr02', 30, 2e-5, 0.5, 0.5),
        # ('test_cps_finetune_t0505_lr03', 20, 3e-5, 0.5, 0.5),
        # ('test_cps_finetune_t0505_lr04', 9, 4e-5, 0.5, 0.5),
        # ('test_cps_finetune_t0505_lr05', 2, 5e-5, 0.5, 0.5),
        # ('test_cps_finetune_t0505_lr10', 1, 1e-4, 0.5, 0.5),
        # ('test_ctd_finetune_lr02_tn_wp_all', 57, 2e-5, None, None),
        # ('test_ctd_finetune_lr02_tn_wp_m1', 41, 2e-5, None, None),
        # ('test_ctd_finetune_lr02_tn_wp_m2', 57, 2e-5, None, None),
        # ('test_ctd_finetune_lr02_tn_wp_m3', 46, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_all', 47, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_m1', 55, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_m2', 41, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_m3', 41, 2e-5, None, None),
        # ('test_ctd_finetune_lr02_tn_rep', 54, 2e-5, None, None),
        # ('test_ctd_finetune_lr02_tn_rep95', 53, 2e-5, None, None),
        # ('test_ctd_finetune_lr02_tn_rep100', 48, 2e-5, None, None),
        # ('test_ctd_finetune_t0705_lr02_rep90', 37, 2e-5, 0.7, 0.99999),
        # ('test_ctd_finetune_t0705_lr02_rep95', 15, 2e-5, 0.7, 0.99999),
        # ('test_ctd_finetune_t0705_lr02_rep100', 58, 2e-5, 0.7, 0.99999),
        # ('test_ctd_finetune_tn_lr02_wp_all_s95', 49, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_m1_s95', 36, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_m2_s95', 49, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_m3_s95', 59, 2e-5, None, None),
        # ('test_ctd_finetune_lr02_tn_wp_all_rep95', 48, 2e-5, None, None),
        # ('test_ctd_finetune_lr02_tn_wp_m1_rep95', 48, 2e-5, None, None),
        # ('test_ctd_finetune_lr02_tn_wp_m2_rep95', 53, 2e-5, None, None),
        # ('test_ctd_finetune_lr02_tn_wp_m3_rep95', 56, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_all_s100', 57, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_m1_s100', 50, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_m2_s100', 55, 2e-5, None, None),
        # ('test_ctd_finetune_tn_lr02_wp_m3_s100', 52, 2e-5, None, None),
        ('test_ctd_finetune_lr02_tn_wp_all_rep100', 56, 2e-5, None, None),
        ('test_ctd_finetune_lr02_tn_wp_m1_rep100', 50, 2e-5, None, None),
        ('test_ctd_finetune_lr02_tn_wp_m2_rep100', 12, 2e-5, None, None),
        ('test_ctd_finetune_lr02_tn_wp_m3_rep100', 43, 2e-5, None, None),
    ]
    for name, epoch, lr, st, tst in test_list:
        print('------', name, epoch, lr, st, tst, '------')
        config_file = open('config/auto.json', 'w', encoding='utf-8')
        config = {
            'model_name': name,
            'learning_rate': lr
        }
        assert not ((st is None) ^ (tst is None))
        if st is not None:
            config['score_threshold'] = st
            config['test_score_threshold'] = tst
            if use_cp:
                config['score_path'] = {
                    'train': f'{rank_result_path}/chemprot_train_sent_fr_score.npy',
                    'valid': f'{rank_result_path}/chemprot_dev_sent_fr_score.npy',
                    'test': f'{rank_result_path}/chemprot_test_sent_fr_score.npy'
                }
                config['title_path'] = {
                    'train': f'{rank_result_path}/chemprot_train_sent_fr_title.json',
                    'valid': f'{rank_result_path}/chemprot_dev_sent_fr_title.json',
                    'test': f'{rank_result_path}/chemprot_test_sent_fr_title.json'
                }
            else:
                config['score_path'] = {
                    'train': f'{rank_result_path}/train_mixed_score.npy',
                    'valid': f'{rank_result_path}/dev_score.npy',
                    'test': f'{rank_result_path}/test_score.npy'
                }
                config['title_path'] = {
                    'train': f'{rank_result_path}/train_mixed_title.json',
                    'valid': f'{rank_result_path}/dev_title.json',
                    'test': f'{rank_result_path}/test_title.json'
                }
        json.dump(config, config_file)
        config_file.close()
        # test step
        print('------', name, epoch, lr, st, tst, '------', file=out_file)
        args = ['python', 'main.py', '-t', 'finetune', '-m', 'test', '-c', f'checkpoint/{name}/model/{epoch}.pkl']
        print('------', ' '.join(args), '------', file=out_file)
        ret = subprocess.Popen(args, stdout=out_train, stderr=out_train)
        assert ret.wait() == 0
        # eval step
        args = ['python', 'eval.py', '-r', f'checkpoint/{name}/test/result.json', '-o',
                f'checkpoint/{name}/test/score.txt']
        print('------', ' '.join(args), '------', file=out_file)
        out_file.flush()
        ret = subprocess.Popen(args, stdout=out_file, stderr=out_file)
        assert ret.wait() == 0
    out_train.close()
    out_file.close()
