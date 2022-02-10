CUDA_ID=0
export CUDA_VISIBLE_DEVICES=$CUDA_ID

TASK=$1
MODEL=microsoft/cocolm-large
LR=1e-5
BS=16
TH=0.8
SM=0.15
MOMENT=0.8
INTERVAL=100
REG=10
RAMP=10
OUT_DIR=result/$TASK

TASK_EXTRA=""
case $TASK in
    CoLA)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{'0':'incorrect','1':'correct'}"
        TASK_EXTRA="--threshold 0"
        ;;
    SST-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{'0':'terrible','1':'great'}"
        ;;
    MRPC)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    QQP)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    MNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        ;;
    QNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        ;;
    RTE)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
esac

for SEED in 13 21 42 87 100
do
    LOG_DIR=$MODEL-seed-$SEED
    python finetune.py \
        --task_name $TASK \
        --data_dir data/$TASK \
        --overwrite_output_dir \
        --do_train \
        --do_predict \
        --smooth $SM \
        --momentum $MOMENT \
        --eval_steps $INTERVAL \
        --threshold $TH \
        --reg_weight $REG \
        --temp_ensemble_rampup $RAMP \
        --model_name_or_path $MODEL \
        --finetune_type prompt \
        --max_seq_length 128 \
        --first_sent_limit 100 \
        --per_device_train_batch_size $BS \
        --gradient_accumulation_steps 1 \
        --learning_rate $LR \
        --num_train_epochs 3 \
        --output_dir $OUT_DIR \
        --seed $SEED \
        --template $TEMPLATE \
        --mapping $MAPPING \
        --logging_dir tb_log/$TASK/$LOG_DIR \
        --logging_steps 20 \
        --warmup_ratio 0.1 \
        --save_at_last \
        $TASK_EXTRA
done
