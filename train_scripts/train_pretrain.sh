export CUDA_VISIBLE_DEVICES=2,3

TGT_FILE=pretrain
MODELFILE=checkpoints/${TGT_FILE}
DATAFILE=/mnt/nvme-data1/waris/PSI-TAMU/DST/data/translation_autoencode/spanish-american/tokenized
PRETRAINED_MODEL=/mnt/nvme-data1/waris/PSI-TAMU/DST/checkpoints/denoise_pretrain/checkpoint_last.pt  # Path to the best pretrain checkpoint

python train.py --ddp-backend=no_c10d ${DATAFILE} --arch transformer_lm \
 --task translation \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --decoder-attention-heads 4 \
 --decoder-layers 8 \
 --criterion decoder_only \
 --report-accuracy \
 --label-smoothing 0.1 \
 --save-dir ${MODELFILE} \
 --restore-file ${PRETRAINED_MODEL} \
 --max-tokens 8192 --update-freq 1 \
 --skip-invalid-size-inputs-valid-test \
 --keep-best-checkpoints 10 \
 --best-checkpoint-metric loss \
 --fp16 \
 --max-target-positions 1024 \
 --tokens-per-sample 1024 \
 --reset-optimizer \
 --reset-lr-scheduler \
 --reset-meters \
 --reset-dataloader \
 --is-pretrain \
 --log-interval 100 > train_log/${TGT_FILE}.txt 2>&1 &