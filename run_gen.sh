export CUDA_VISIBLE_DEVICES=0

TASK=$1
LABEL=all
SAVE_DIR=temp_gen

case $TASK in
    MNLI)
        NUM_GEN=25000
        MAX_LEN=40
        TEMP=0
        TASK_EXTRA="--pretrain_corpus_dir pretrain_corpus/wiki_short.txt"
        ;;
    QQP)
        NUM_GEN=25000
        MAX_LEN=50
        TEMP=0
        TASK_EXTRA="--pretrain_corpus_dir pretrain_corpus/openwebtext_questions.txt"
        ;;
    QNLI)
        NUM_GEN=25000
        MAX_LEN=100
        TEMP=0
        TASK_EXTRA="--pretrain_corpus_dir pretrain_corpus/openwebtext_questions.txt"
        ;;
    SST-2)
        NUM_GEN=25000
        MAX_LEN=40
        TEMP=0.2
        ;;
    CoLA)
        NUM_GEN=20000
        MAX_LEN=40
        TEMP=[0.1,10]
        ;;
    RTE)
        NUM_GEN=30000
        MAX_LEN=40
        TEMP=0
        TASK_EXTRA="--pretrain_corpus_dir pretrain_corpus/wiki_long.txt"
        ;;
    MRPC)
        NUM_GEN=30000
        MAX_LEN=40
        TEMP=0
        TASK_EXTRA="--pretrain_corpus_dir pretrain_corpus/wiki_long.txt"
        ;;
esac

# Generate training data
python gen_train_data.py --task $TASK --label $LABEL --save_dir $SAVE_DIR --print_res \
                         --num_gen $NUM_GEN --max_len $MAX_LEN --temperature $TEMP \
                         $TASK_EXTRA

# Select training data
DATA_DIR=data/${TASK}
NUM_SELECT=6000
python src/gen_utils.py --task $TASK --num_select_samples $NUM_SELECT \
                        --read_dir $SAVE_DIR --save_dir $DATA_DIR
