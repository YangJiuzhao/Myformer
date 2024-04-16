############################predict length 24####################################
python -u main_former.py --data ETTm1 \
--in_len 288 --out_len 24 --seg_len 3,6,12,24 \
--learning_rate 1e-4 --itr 5 >logs/crossformer_ETTm1_24.log

############################predict length 48####################################
python -u main_former.py --data ETTm1 \
--in_len 288 --out_len 48 --seg_len 6,8,12,24,36,48 \
--learning_rate 1e-4 --itr 5 >logs/crossformer_ETTm1_48.log

############################predict length 96####################################
python -u main_former.py --data ETTm1 \
--in_len 672 --out_len 96 --seg_len 12,16,24,48,96 \
--learning_rate 1e-4 --itr 5 >logs/crossformer_ETTm1_96.log

############################predict length 288####################################
python -u main_former.py --data ETTm1 \
--in_len 672 --out_len 288 --seg_len 24,36,48,72,96 \
--learning_rate 1e-5 --itr 5 >logs/crossformer_ETTm1_288.log

############################predict length 672####################################
python -u main_former.py --data ETTm1 \
--in_len 672 --out_len 672 --seg_len 12,24,36,48,72,96 \
--learning_rate 1e-5 --itr 5 >logs/crossformer_ETTm1_672.log