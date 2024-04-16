############################predict length 30####################################
python -u main_former.py --data m_1999 --model lstm \
--in_len 300 --out_len 30 --seg_len 6 \
--learning_rate 1e-3 --itr 1 >logs/415/lstm_m_1999_30.log

############################predict length 60####################################
python -u main_former.py --data m_1999 --model lstm \
--in_len 300 --out_len 60 --seg_len 6 \
--learning_rate 1e-3 --itr 1 >logs/415/lstm_m_1999_60.log

############################predict length 90####################################
python -u main_former.py --data m_1999 --model lstm \
--in_len 300 --out_len 90 --seg_len 9 \
--learning_rate 1e-3 --itr 1 >logs/415/lstm_m_1999_90.log

############################predict length 180####################################
python -u main_former.py --data m_1999 --model lstm \
--in_len 300 --out_len 180 --seg_len 30 \
--learning_rate 1e-3 --itr 1 >logs/415/lstm_m_1999_180.log

############################predict length 360####################################
python -u main_former.py --data m_1999 --model lstm \
--in_len 300 --out_len 360 --seg_len 30 \
--learning_rate 1e-3 --itr 1 >logs/415/lstm_m_1999_360.log

############################predict length 720####################################
python -u main_former.py --data m_1999 --model lstm \
--in_len 300 --out_len 720 --seg_len 30 \
--learning_rate 1e-3 --itr 1 >logs/415/lstm_m_1999_720.log