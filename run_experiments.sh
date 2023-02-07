for MIN_LR in 0.0
do
    for EPOCH in 100 300 500
    do
        for BATCH_SIZE in 4 8 12 16
        do
            CUDA_VISIBLE_DEVICES=0,1,2,3 python train_val_transfer.py --scheduler_type 'CosineLR' --min_lr $MIN_LR --batch_size $BATCH_SIZE --total_epoch $EPOCH --feature_list 0 1 2 3 4 5 6
        done
    done
done
