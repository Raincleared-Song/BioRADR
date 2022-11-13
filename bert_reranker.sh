# downloaded from the link mentioned in 'Data Availability'
cd OpenMatch
# generate BERT Re-Ranker training set
python build_train.py \
    --tokenizer_name allenai/scibert_scivocab_cased \
    --sample_file ../trec_pm/train_triplets_ctdred_1032.jsonl \
    --queries ../trec_pm/queries_ctdred.tsv \
    --queries_cid ../trec_pm/queries_cid_ctdred.tsv \
    --re_train_file ../CTDRED/ctd_train.json \
    --pmid_to_did ../trec_pm/pmid_to_did.json \
    --entity_marker true \
    --intra_priority false \
    --save_to ../trec_pm/scibert_ctdred_1032_marker \
    --doc_template "<title> <text> [SEP]" \
    --query_template "<text>" \
    --doc_max_len 478 \
    --query_max_len 32
# train BERT Re-Ranker
export CUDA_VISIBLE_DEVICES=0  # gpu index can be changed
python train_rerank.py  \
    --output_dir checkpoints/scibert_ctdred_1032_marker  \
    --model_name_or_path allenai/scibert_scivocab_cased  \
    --do_train  \
    --save_steps 500  \
    --train_path ../trec_pm/scibert_ctdred_1032_marker/train_open.jsonl  \
    --per_device_train_batch_size 16  \
    --learning_rate 1e-5  \
    --q_max_len 32  \
    --p_max_len 478  \
    --num_train_epochs 10  \
    --loss_fn bce  \
    --logging_dir checkpoints/scibert_ctdred_log  \
    --dataloader_num_workers 1 \
    --seed 100
# validation and test
python eval_openmatch.py -t scibert_ctdred_1032_marker --entity_marker

# generate BERT Re-Ranker+ training set
python build_train.py \
    --tokenizer_name allenai/scibert_scivocab_cased \
    --sample_file ../trec_pm/train_triplets_ctdred_1032.jsonl \
    --queries ../trec_pm/queries_ctdred.tsv \
    --queries_cid ../trec_pm/queries_cid_ctdred.tsv \
    --re_train_file ../CTDRED/ctd_train.json \
    --pmid_to_did ../trec_pm/pmid_to_did.json \
    --entity_marker true \
    --intra_priority true \
    --save_to ../trec_pm/scibert_ctdred_1032_intra \
    --doc_template "<title> <text> [SEP]" \
    --query_template "<text>" \
    --doc_max_len 478 \
    --query_max_len 32
# train BERT Re-Ranker+
export CUDA_VISIBLE_DEVICES=0  # gpu index can be changed
python train_rerank.py  \
    --output_dir checkpoints/scibert_ctdred_1032_intra  \
    --model_name_or_path allenai/scibert_scivocab_cased  \
    --do_train  \
    --save_steps 500  \
    --train_path ../trec_pm/scibert_ctdred_1032_intra/train_open.jsonl  \
    --per_device_train_batch_size 16  \
    --learning_rate 1e-5  \
    --q_max_len 32  \
    --p_max_len 478  \
    --num_train_epochs 10  \
    --loss_fn bce  \
    --logging_dir checkpoints/scibert_ctdred_log  \
    --dataloader_num_workers 1 \
    --seed 100
# validation and test
python eval_openmatch.py -t scibert_ctdred_1032_intra --entity_marker --intra_priority
