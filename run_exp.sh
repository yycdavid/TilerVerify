#OFFSET_RANGE_TRAIN=50
#ANGLE_RANGE_TRAIN=70
noise_mode=uniform
noise_scale=0.01
#python generate_data.py --mode train --offset_range $OFFSET_RANGE_TRAIN --angle_range $ANGLE_RANGE_TRAIN --noise $noise_mode --#noise_scale $noise_scale --target_dir_name train_"$noise_mode"_"$noise_scale"

#TRAIN_DATA=train_bigger_130000.mat
#VALID_DATA=valid_bigger_1000.mat
#RESULT_FOLDER="$noise_mode"_"$noise_scale"
#python trainer/train.py --case road --dataset_folder train_"$noise_mode"_"$noise_scale" --train_data $TRAIN_DATA --val_data $VALID_DATA --result $RESULT_FOLDER
#python trainer/convert_for_milp.py --case road --name $RESULT_FOLDER


OFFSET_RANGE=40
ANGLE_RANGE=60
num_threads=20
exp_name="$noise_mode"_"$noise_scale"
for grid_size in 0.1
do
    #export JULIA_NUM_THREADS=$num_threads
    # Generate dataset for estimate and bound
    python parallel_verify.py --offset_range $OFFSET_RANGE --angle_range $ANGLE_RANGE --grid_size $grid_size --num_threads $num_threads

    # Compute bound by verify
    data_name=verify_offset_"$OFFSET_RANGE"_angle_"$ANGLE_RANGE"_grid_"$grid_size"_thread_"$num_threads"none0.05
    for thread_number in $(seq 0 $(expr $num_threads - 1))
    do
        /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia verify_thread.jl $exp_name $data_name $thread_number &
        sleep .5
    done
    wait
    /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia thread_collect.jl $data_name $num_threads
    # Compute estimate
    python generate_data.py --mode estimate --offset_range $OFFSET_RANGE --angle_range $ANGLE_RANGE --grid_size $grid_size --target_dir_name $data_name
    python trainer/error_estimate.py --exp_name $exp_name --target_dir_name $data_name --grid_size $grid_size
    # Get heatmap
    python analysis/heatmap.py --result_dir data/"$data_name" --offset_range $OFFSET_RANGE --angle_range $ANGLE_RANGE
    python analysis/statistics.py --result_dir data/"$data_name"
done
