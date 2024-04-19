############################predict length 30####################################
python -u main_former.py --data m1999m --model lstm --baseline True \
--in_len 360 --out_len 30 --seg_len 6 \
--learning_rate 1e-4 --itr 1 >logs/418/lstm_m1999m_360_30.log

############################predict length 60####################################
python -u main_former.py --data m1999m --model lstm --baseline True \
--in_len 360 --out_len 60 --seg_len 6 \
--learning_rate 1e-4 --itr 1 >logs/418/lstm_m1999m_360_60.log

############################predict length 90####################################
python -u main_former.py --data m1999m --model lstm --baseline True \
--in_len 360 --out_len 90 --seg_len 9 \
--learning_rate 1e-4 --itr 1 >logs/418/lstm_m1999m_360_90.log

############################predict length 180####################################
python -u main_former.py --data m1999m --model lstm --baseline True \
--in_len 360 --out_len 180 --seg_len 30 \
--learning_rate 1e-4 --itr 1 >logs/418/lstm_m1999m_360_180.log

############################predict length 360####################################
python -u main_former.py --data m1999m --model lstm --baseline True \
--in_len 360 --out_len 360 --seg_len 30 \
--learning_rate 1e-4 --itr 1 >logs/418/lstm_m1999m_360_360.log

############################predict length 480####################################
python -u main_former.py --data m1999m --model lstm --baseline True \
--in_len 360 --out_len 480 --seg_len 30 \
--learning_rate 1e-4 --itr 1 >logs/418/lstm_m1999m_360_480.log

############################predict length 600####################################
python -u main_former.py --data m1999m --model lstm --baseline True \
--in_len 360 --out_len 600 --seg_len 30 \
--learning_rate 1e-4 --itr 1 >logs/418/lstm_m1999m_360_600.log

############################predict length 720####################################
python -u main_former.py --data m1999m --model lstm --baseline True \
--in_len 360 --out_len 720 --seg_len 30 \
--learning_rate 1e-4 --itr 1 >logs/418/lstm_m1999m_360_720.log