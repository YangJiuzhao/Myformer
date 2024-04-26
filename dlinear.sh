if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/426" ]; then
    mkdir ./logs/426
fi
in_len=336
model_name=patchtst

model_id_name=ETTm1
data_name=ETTm1

random_seed=2024
for out_len in 96 192 336 720
do
    python -u main_former.py \
        --baseline True \
        --model $model_name \
        --data $data_name \
        --in_len $in_len \
        --out_len $out_len \
        --a_layers 3 \
        --n_heads 4 \
        --d_model 64 \
        --d_ff 128 \
        --itr 1 \
        --batch_size 32 \
        --dropout 0.3 \
        --gpu 1 \
        --learning_rate 0.0001 >logs/426/$model_name'_'$model_id_name'_'$in_len'_'$out_len.log 
done