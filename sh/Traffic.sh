############################predict length 24####################################
python main_former.py --data Traffic \
--in_len 96 --out_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 1e-3  --itr 1 >logs/Traffic24.log

############################predict length 48####################################
python main_former.py --data Traffic \
--in_len 96 --out_len 48 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 1e-3 --itr 1 >logs/Traffic48.log

############################predict length 168####################################
python main_former.py --data Traffic \
--in_len 336 --out_len 168 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 1e-3 --itr 1 >logs/Traffic168.log

############################predict length 336####################################
python main_former.py --data Traffic \
--in_len 720 --out_len 336 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 1e-3 --itr 1 >logs/Traffic336.log

############################predict length 720####################################
python main_former.py --data Traffic \
--in_len 336 --out_len 720 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-3 --itr 1 >logs/Traffic720.log