# BioRADR

Source code for paper _BioRADR: Biomedical Relation-Aware Document Ranking_.

### Environment
The anaconda environment is provided in `conda_env.yaml`.

### Data Availability
Download data from [link](https://drive.google.com/drive/folders/1DQHSG2tdHLQt6uXfhWXfqRD_wXE14uoq?usp=sharing) and put those files under the project directory.

### Performance in Document-Level RE

#### Pre-Training on CTDRED
You should first modify `config/config_base.py` and `config/config_finetune.py` according to `config_bak/docre_pretrain.py`, and execute:
```bash
python3 main.py -t finetune -m train
```
Then adopt checkpoint epoch-0 and abandon others.

#### Fine-Tuning on BC5CDR
To reproduce the results, you should first change `config/config_base.py` and `config/config_finetune.py` according to `config_bak/docre_docunet.py` (DocuNet+Tuned) and `config_bak/docre_biodre.py` (BioRADR). Then, modify the line 51 of `main.py` to `is_cdr = True` for automatic 5-time replication.

Then, for training:
```bash
python3 main.py -t finetune -m train -c {pre-trained checkpoint path, omissible}
```

For testing, modify the line 51 of `main.py` to `is_cdr = False`, and then:
```bash
python3 batch_test_cdr.py -t {model name under the dir 'checkpoint'}
```

### Performance in BioRADR
#### Training on CTDRED
You should first modify `config/config_base.py` and `config/config_denoise.py` according to `config_bak/rar_docunet.py` (DocuNet+Tuned) and `config_bak/rar_biodre.py` (BioRADR).
```bash
python3 main.py -t denoise -m train
```
#### Testing on the Golden BioRADR Test Set
After training, do validation on group 0,1,25,26 to choose the best model (k=-1 for NDCG@All):
```bash
python3 test_ndcg.py -d manual/rank_files/0_pmc_segments_sample.json,\
manual/rank_files/1_pmc_segments_sample.json,manual/rank_files/25_pmc_segments_sample.json,manual/rank_files/26_pmc_segments_sample.json \
-a manual/manual_new/0_seg_ans.txt,manual/manual_new/1_seg_ans.txt,manual/manual_new/25_seg_ans.txt,manual/manual_new/26_seg_ans.txt \
-p {parent dir of checkpoints to be tested} \
-rp CTDRED/temp_range -mn {model name under the dir 'checkpoint'} -n {NDCG@k: -1,50,20,10,5,1}
```
Then, for testing:
```bash
python3 test_ndcg.py -d manual/rank_files -a manual/manual_new \
-p {the chosen checkpoint path} \
-rp CTDRED/temp_range -mn {model name under the dir 'checkpoint'} -i 0,1,25,26 -n {NDCG@k: -1,50,20,10,5,1}
```
For results on Inside-CTD groups and Outside-CTD groups, change the option `-i 0,1,25,26` to `-i 0,1,25-49` and `0-24,25,26` respectively. To obtain results of `BM25` and `PMC`, run the following script:
```bash
python3 test_ranking.py
```
