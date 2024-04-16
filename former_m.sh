############################predict length 30####################################
python -u main_former.py --data m1999m \
--in_len 288 --out_len 24 --seg_len 3,6,12,24 \
--learning_rate 1e-4 --itr 1 >logs/416/former_m1999m_24.log

############################predict length 60####################################
python -u main_former.py --data m1999m \
--in_len 288 --out_len 48 --seg_len 3,6,12,24,48 \
--learning_rate 1e-4 --itr 1 >logs/416/former_m1999m_48.log

############################predict length 90####################################
python -u main_former.py --data m1999m \
--in_len 672 --out_len 96 --seg_len 12,24,36,48 \
--learning_rate 1e-4 --itr 1 >logs/416/former_m1999m_96.log

############################predict length 180####################################
python -u main_former.py --data m1999m \
--in_len 672 --out_len 288 --seg_len 12,24,36,48,72 \
--learning_rate 1e-4 --itr 1 >logs/416/former_m1999m_288.log

############################predict length 360####################################
python -u main_former.py --data m1999m \
--in_len 672 --out_len 672 --seg_len 12,24,36,48,72 \
--learning_rate 1e-4 --itr 1 >logs/416/former_m1999m_672.log

############################predict length 720####################################
# python -u main_former.py --data m_1999 \
# --in_len 300 --out_len 720 --seg_len 30,60,90,120,180 \
# --learning_rate 1e-4 --itr 1 >logs/415/former_m1999_720.log