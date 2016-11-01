#!/bin/bash
#declare -a arr=("G1" "G60" "G81" "dbn11" "protein_p_1" "protein_p_2" "grid80")
declare -a arr=("G1" "grid80"  "G60" "G81" "dbn11" "protein_p_1" "protein_p_2")
#echo "" > ../results/gdmm_time.txt
for dataset in "${arr[@]}"
do
	echo $dataset
	./sdp_omp -p 1 -e 100 -t 1 -s 1e-8  -i 10 -o 1 -y 10000 ../../../repeat_exp/data/$dataset > ../../../repeat_exp/results/rank_k5_$dataset.txt
#	./sdp_omp -p 1 -e 100 -t 1 -s 1e-6  -i 10 -y 10000 ../data/$dataset >> ../results/rank_$dataset.txt
done
