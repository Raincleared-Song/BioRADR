import json
import subprocess


def tune_test_threshold(mode):
    use_cp = True
    if use_cp:
        out_file = open('test_cps_tune_thred_log.txt', 'a', encoding='utf-8')
        out_train = open('test_cps_train_thred_log.txt', 'a', encoding='utf-8')
        rank_result_path = 'Chemprot/rank_result_fr_new_5'
    else:
        out_file = open('test_tune_thred_log.txt', 'a', encoding='utf-8')
        out_train = open('test_train_thred_log.txt', 'a', encoding='utf-8')
        rank_result_path = 'CTDRED/rank_result'
    tune_list = [None]
    name = 'test_cps_finetune_new_lr03'
    epoch = 10
    lr = 3e-5
    st = None
    for tst in tune_list:
        print('------', name, epoch, lr, st, tst, mode, '------')
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
                    'test': f'{rank_result_path}/chemprot_{"dev" if mode == "valid" else "test"}_sent_fr_score.npy'
                }
                config['title_path'] = {
                    'train': f'{rank_result_path}/chemprot_train_sent_fr_title.json',
                    'valid': f'{rank_result_path}/chemprot_dev_sent_fr_title.json',
                    'test': f'{rank_result_path}/chemprot_{"dev" if mode == "valid" else "test"}_sent_fr_title.json'
                }
            else:
                config['score_path'] = {
                    'train': f'{rank_result_path}/train_mixed_score.npy',
                    'valid': f'{rank_result_path}/dev_score.npy',
                    'test': f'{rank_result_path}/{"dev" if mode == "valid" else "test"}_score.npy'
                }
                config['title_path'] = {
                    'train': f'{rank_result_path}/train_mixed_title.json',
                    'valid': f'{rank_result_path}/dev_title.json',
                    'test': f'{rank_result_path}/{"dev" if mode == "valid" else "test"}_title.json'
                }
        # 只在验证集上测试调参
        if use_cp:
            config['data_path'] = {
                'train': 'Chemprot/chemprot_train_sent_fr.json',
                'valid': 'Chemprot/chemprot_dev_sent_fr.json',
                'test': f'Chemprot/chemprot_{"dev" if mode == "valid" else "test"}_sent_fr.json'
            }
        else:
            config['data_path'] = {
                'train': 'CTDRED/train_mixed.json',
                'valid': 'CTDRED/dev.json',
                'test': f'CTDRED/{"dev" if mode == "valid" else "test"}.json'
            }
        json.dump(config, config_file)
        config_file.close()
        # test step
        print('------', name, epoch, lr, st, tst, mode, '------', file=out_file)
        args = ['python', 'main.py', '-t', 'finetune', '-m', 'test', '-c', f'checkpoint/{name}/model/{epoch}.pkl']
        print('------', ' '.join(args), '------', file=out_file)
        ret = subprocess.Popen(args, stdout=out_train, stderr=out_train)
        assert ret.wait() == 0
        # eval step
        args = ['python', 'eval.py', '-r', f'checkpoint/{name}/test/result.json', '-o',
                f'checkpoint/{name}/test/score_{tst}_{mode}.txt']
        if mode == 'valid':
            args.append('-d')
        print('------', ' '.join(args), '------', file=out_file)
        out_file.flush()
        ret = subprocess.Popen(args, stdout=out_file, stderr=out_file)
        assert ret.wait() == 0
    out_train.close()
    out_file.close()


if __name__ == '__main__':
    tune_test_threshold('valid')
    # tune_test_threshold('test')
