############################predict length 30####################################
python -u main_former.py --data m1999m \
--in_len 288 --out_len 24 --seg_len 3,6,10,15,30 \
--learning_rate 1e-4 --itr 1 >logs/415/former_m1999_30.log

############################predict length 60####################################
python -u main_former.py --data m_1999 \
--in_len 300 --out_len 60 --seg_len 6,10,15,20,30,60 \
--learning_rate 1e-4 --itr 1 >logs/415/former_m1999_60.log

############################predict length 90####################################
python -u main_former.py --data m_1999 \
--in_len 300 --out_len 90 --seg_len 6,10,15,30,45,90 \
--learning_rate 1e-5 --itr 1 >logs/415/former_m1999_90.log

############################predict length 180####################################
python -u main_former.py --data m_1999 \
--in_len 300 --out_len 180 --seg_len 30,45,60,90,180 \
--learning_rate 1e-5 --itr 1 >logs/415/former_m1999_180.log

############################predict length 360####################################
python -u main_former.py --data m_1999 \
--in_len 300 --out_len 360 --seg_len 30,40,60,90,120,180 \
--learning_rate 1e-5 --itr 1 >logs/415/former_m1999_360.log

############################predict length 720####################################
python -u main_former.py --data m_1999 \
--in_len 300 --out_len 720 --seg_len 30,60,90,120,180 \
--learning_rate 1e-4 --itr 1 >logs/415/former_m1999_720.log