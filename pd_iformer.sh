if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/426" ]; then
    mkdir ./logs/426
fi
in_len=336
model_name=iformer
seg_len=64,32,24,12,8,6
# 64,48,32,24,12,8,6
model_id_name=ETTm1
data_name=ETTm1

random_seed=2024
for out_len in 96 192 336 720
do
    python -u main_former.py \
        --model $model_name \
        --data $data_name \
        --in_len $in_len \
        --out_len $out_len \
        --seg_len $seg_len \
        --a_layers 3 \
        --n_heads 4 \
        --d_model 64 \
        --d_ff 128 \
        --itr 1 \
        --batch_size 32 \
        --dropout 0.4 \
        --gpu 0 \
        --learning_rate 0.0001 >logs/426/$model_name'_'$model_id_name'_'$in_len'_'$out_len.log 
done