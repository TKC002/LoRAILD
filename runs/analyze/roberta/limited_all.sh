# bash /work/GLUE-KD/runs/analyze/roberta/KD/limited_all.sh
# bash /work/GLUE-KD/runs/analyze/roberta/CurriculumFullLoRAILD_new/limited_all.sh
# bash /work/GLUE-KD/runs/analyze/roberta/RAILKD_new/limited_all.sh

for i in `seq 1 5` ; do
bash runs/analyze/roberta/limited_$i.sh
python cluster.py confs/analyze/roberta/clustering/all$i.yaml
rm -r /work2/GLUE-KD/analyze/roberta/CurriculumFullLoRAILD
rm -r /work2/GLUE-KD/analyze/roberta/FullLoRALoRAKD
rm -r /work2/GLUE-KD/analyze/roberta/RAILKD_l
done