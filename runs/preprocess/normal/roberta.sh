tasks=(cola mnli mrpc qnli qqp rte sst2 stsb wnli)

for task in ${tasks[@]}
do
    python normal_preprocess.py --task $task --tokenizer roberta-large --max_length 128
done