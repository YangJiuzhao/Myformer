############################predict length 24####################################
python -u main_former.py --data ETTh1 \
--in_len 168 --out_len 24 --seg_lens 3,6,12,24 \
--learning_rate 1e-4 --itr 5 >logs/ETTh1_24.log

############################predict length 48####################################
python -u main_former.py --data ETTh1 \
--in_len 168 --out_len 48 --seg_lens 6,12,24,36,48 \
--learning_rate 1e-4 --itr 5 >logs/ETTh1_48.log

############################predict length 168###################################
python -u main_former.py --data ETTh1  \
--in_len 720 --out_len 168 --seg_lens 6,12,24,36,48,72 \
--learning_rate 1e-5 --itr 5 >logs/ETTh1_168.log

############################predict length 336###################################
python -u main_former.py --data ETTh1 \
--in_len 720 --out_len 336 --seg_lens 6,12,24,36,48,72 \
--learning_rate 1e-5 --itr 5 >logs/ETTh1_336.log

############################predict length 720###################################
python -u main_former.py --data ETTh1 \
--in_len 720 --out_len 720 --seg_lens 6,12,24,36,48,72 \
--learning_rate 1e-5 --itr 5 >logs/ETTh1_720.log