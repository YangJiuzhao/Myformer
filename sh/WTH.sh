############################predict length 24####################################
python -u main_former.py --data WTH \
--in_len 48 --out_len 24 --seg_len 6,8,12 \
--learning_rate 1e-4 --itr 5 >logs/WTH_24.log

############################predict length 24####################################
python -u main_former.py --data WTH \
--in_len 48 --out_len 48 --seg_len 12,16,24 \
--learning_rate 1e-4 --itr 5 >logs/WTH_48.log

############################predict length 168###################################
python -u main_former.py --data WTH \
--in_len 336 --out_len 168 --seg_len 24,36,48,56 \
--learning_rate 1e-5 --itr 5 >logs/WTH_168.log

############################predict length 336###################################
python -u main_former.py --data WTH \
--in_len 336 --out_len 336 --seg_len 24,36,48,56 \
--learning_rate 1e-5 --itr 5 >logs/WTH_336.log

############################predict length 720###################################
python -u main_former.py --data WTH \
--in_len 720 --out_len 720 --seg_len 24,36,48,72 \
--learning_rate 1e-5 --itr 5 >logs/WTH_720.log