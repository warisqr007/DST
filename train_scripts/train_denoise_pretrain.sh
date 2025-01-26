export CUDA_VISIBLE_DEVICES=1,2

TGT_FILE=denoise_pretrain
MODELFILE=/mnt/data2/waris/code/PSI-TAMU/DST/checkpoints/${TGT_FILE}
DATAFILE=/mnt/nvme-data1/waris/PSI-TAMU/DST/data/libri_denoising_dedup/tokenized/american

python train.py --ddp-backend=no_c10d ${DATAFILE} \
 --arch transformer_lm \
 --task denoising \
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
 --max-tokens 8192 --update-freq 1 \
 --skip-invalid-size-inputs-valid-test \
 --keep-best-checkpoints 10 \
 --best-checkpoint-metric loss \
 --fp16 \
 --max-target-positions 1024 \
 --tokens-per-sample 1024 \
 --poisson-lambda 3.5 \
 --mask 0.3 \
 --mask-length span-poisson \
 --replace-length 1 \
 --rotate 0 \
 --mask-random 0.1 \
 --insert 0 \
 --permute-sentences 1.0 \
 --is-pretrain \
 --tensorboard-logdir /mnt/data2/waris/code/PSI-TAMU/DST/logs/${TGT_FILE} \
 --log-interval 100 > /mnt/data2/waris/code/PSI-TAMU/DST/train_log/${TGT_FILE}.txt 2>&1 &