if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/418" ]; then
    mkdir ./logs/418
fi
seq_len=360
model_name=dlinear

model_id_name=m1999m
data_name=m1999m

random_seed=2024
for pred_len in 30 60 90 180 360 480 600 720
do
    python -u main_former.py \
        --model $model_name \
        --baseline True \
        --in_len $seq_len \
        --out_len $pred_len \
        --itr 1 \
        --batch_size 32 \
        --learning_rate 0.0001 >logs/418/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done