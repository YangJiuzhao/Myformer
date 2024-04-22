if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/418" ]; then
    mkdir ./logs/418
fi
seq_len=360
model_name=Informer

root_path_name=~/yjz/Myformer/datasets/
data_path_name=m1999m.csv
test_id_name=add


############################predict length 30####################################
python -u main_former.py --data m1999m --model iformer \
--in_len 360 --out_len 30 --seg_len 120,60,36,24,12 \
--learning_rate 1e-4 --itr 1 --dropout 0.4 --gpu 1 >logs/418/iformer_$test_id_name'_m1999m_360_30.log'

############################predict length 60####################################
python -u main_former.py --data m1999m --model iformer \
--in_len 360 --out_len 60 --seg_len 60,36,24,18,12,6 \
--learning_rate 1e-4 --itr 1 --dropout 0.4 --gpu 1 >logs/418/iformer_$test_id_name'_m1999m_360_60.log'

############################predict length 90####################################
python -u main_former.py --data m1999m --model iformer \
--in_len 360 --out_len 90 --seg_len 60,36,24,12,8,4 \
--learning_rate 1e-4 --itr 1 --dropout 0.4 --gpu 1 >logs/418/iformer_$test_id_name'_m1999m_360_90.log'

############################predict length 180####################################
python -u main_former.py --data m1999m --model iformer \
--in_len 360 --out_len 180 --seg_len 12,8,6,4,3 \
--learning_rate 1e-4 --itr 1 --dropout 0.4 --gpu 1 >logs/418/iformer_$test_id_name'_m1999m_360_180.log'

############################predict length 360####################################
python -u main_former.py --data m1999m --model iformer \
--in_len 360 --out_len 360 --seg_len 12,8,6,4,3 \
--learning_rate 1e-5 --itr 1 --dropout 0.4 --gpu 1 >logs/418/iformer_$test_id_name'_m1999m_360_360.log'

############################predict length 480####################################
python -u main_former.py --data m1999m --model iformer \
--in_len 360 --out_len 480 --seg_len 12,8,6,4,3 \
--learning_rate 1e-5 --itr 1 --dropout 0.4 --gpu 1 >logs/418/iformer_$test_id_name'_m1999m_360_480.log'

############################predict length 600####################################
python -u main_former.py --data m1999m \
--in_len 360 --out_len 600 --seg_len 12,8,6,4,3 \
--learning_rate 1e-5 --itr 1 --dropout 0.4 --gpu 1 >logs/418/iformer_$test_id_name'_m1999m_360_600.log'

############################predict length 720####################################
python -u main_former.py --data m1999m --model iformer \
--in_len 360 --out_len 720 --seg_len 12,8,6,4,3 \
--learning_rate 1e-5 --itr 1 --dropout 0.4 --gpu 1 >logs/418/iformer_$test_id_name'_m1999m_360_720.log'