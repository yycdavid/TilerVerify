SAVE_FOLDER=bnn

# Train model
#CUDA_VISIBLE_DEVICES=1 python -m eevbnn train_bin RoadSign0 trained_models/lidar_quant_061_small --input-quant 0.61 --set-global-param g_channel_scale=1 --set-global g_weight_decay=2e-6 --epoch 100

for grid_size in 1.0
do
    DATA_FOLDER=lidar_distance_min_30_max_60_angle_45_grid_"$grid_size"_thread_21gaussian0.001_small
    num_threads_per_class=7

    for shape in 0 1 2
    do
        for thread_number in $(seq 0 $(expr $num_threads_per_class - 1))
        do
            python -m eevbnn eval_bin -e 0.3 --timeout 10 trained_models/lidar_quant_061_small/last.pth --check-cvt --lidar --shape $shape --thread $thread_number --lidar-data-folder $DATA_FOLDER --save-folder $SAVE_FOLDER &
            sleep .5
        done
    done
    wait

    python collect_result_lidar.py --exp_dir $DATA_FOLDER --save-folder $SAVE_FOLDER --num_threads $num_threads_per_class

    python analysis/heatmap_lidar.py --result_dir data/"$DATA_FOLDER"/"$SAVE_FOLDER" --distance_min 30 --distance_max 60 --angle_range 45 --sio
done
