############################predict length 24####################################
python -u main_former.py --data ILI \
--in_len 48 --out_len 24  --seg_len 6,8 \
--a_layers 2 \
--d_model 128 --d_ff 256 --n_heads 2 \
--learning_rate 1e-4 --dropout 0.6 --itr 5 >logs/ILI_24.log

############################predict length 36####################################
python -u main_former.py --data ILI \
--in_len 48 --out_len 36  --seg_len 6,8 \
--a_layers 2 \
--d_model 128 --d_ff 256 --n_heads 2 \
--learning_rate 1e-4 --dropout 0.6 --itr 5 >logs/ILI_36.log

############################predict length 48####################################
python -u main_former.py --data ILI \
--in_len 60 --out_len 48  --seg_len 6,8 \
--a_layers 2 \
--d_model 128 --d_ff 256 --n_heads 2 \
--learning_rate 1e-4 --dropout 0.6 --itr 5 >logs/ILI_48.log

############################predict length 60####################################
python -u main_former.py --data ILI \
--in_len 60 --out_len 60  --seg_len 6,8 \
--a_layers 2 \
--d_model 128 --d_ff 256 --n_heads 2 \
--learning_rate 1e-4 --dropout 0.6 --itr 5 >logs/ILI_60.log