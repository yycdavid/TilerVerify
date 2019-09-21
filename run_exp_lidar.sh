DISTANCE_MIN_TRAIN=15
DISTANCE_MIN_VALID=20
DISTANCE_MAX_TRAIN=65
DISTANCE_MAX_VALID=60
ANGLE_RANGE_TRAIN=50
ANGLE_RANGE_VALID=45
noise_mode=gaussian
noise_scale=0.001
#python3 generate_data_lidar.py --mode train --distance_max_train $DISTANCE_MAX_TRAIN --distance_min_train $DISTANCE_MIN_TRAIN --angle_range_train $ANGLE_RANGE_TRAIN --distance_max_valid $DISTANCE_MAX_VALID --distance_min_valid $DISTANCE_MIN_VALID --angle_range_valid $ANGLE_RANGE_VALID --noise $noise_mode --noise_scale $noise_scale --target_dir_name lidar_train_"$noise_mode"_"$noise_scale"

TRAIN_DATA=train.pickle
VALID_DATA=valid.pickle
RESULT_FOLDER=lidar_"$noise_mode"_"$noise_scale"
#python3 trainer/train.py --case lidar --dataset_folder lidar_train_"$noise_mode"_"$noise_scale" --train_data $TRAIN_DATA --val_data $VALID_DATA --result $RESULT_FOLDER --cuda
#python3 trainer/convert_for_milp.py --case lidar --name $RESULT_FOLDER

DISTANCE_MIN=20
DISTANCE_MAX=60
ANGLE_RANGE=45
num_threads=21
#exp_name="$noise_mode"_"$noise_scale"
exp_name=$RESULT_FOLDER
for grid_size in 1.0
do
    export JULIA_NUM_THREADS=$num_threads
    # Generate dataset for estimate and bound
    python3 parallel_verify_lidar.py --distance_min $DISTANCE_MIN --distance_max $DISTANCE_MAX --angle_range $ANGLE_RANGE --grid_size $grid_size --num_threads $num_threads --noise $noise_mode --noise_scale $noise_scale

    # Compute bound by verify
    #data_name=lidar_distance_min_"$DISTANCE_MIN"_max_"$DISTANCE_MAX"_angle_"$ANGLE_RANGE"_grid_"$grid_size"_thread_"$num_threads""$noise_mode""$noise_scale"
    #for thread_number in $(seq 0 $(expr $num_threads - 1))
    #do
    #    /raid/yicheny/software/julia-9d11f62bcb/bin/julia verify_thread.jl $exp_name $data_name $thread_number &
    #    sleep .5
    #done
    #wait
    #/raid/yicheny/software/julia-9d11f62bcb/bin/julia thread_collect.jl $data_name $num_threads
    # Compute estimate
    #python3 generate_data.py --mode estimate --offset_range $OFFSET_RANGE --angle_range $ANGLE_RANGE --grid_size $grid_size #--arget_dir_name $data_name --noise $noise_mode --noise_scale $noise_scale
    #python3 trainer/error_estimate.py --exp_name $exp_name --target_dir_name $data_name --grid_size $grid_size
    # Get heatmap
    #python3 analysis/heatmap.py --result_dir data/"$data_name" --offset_range $OFFSET_RANGE --angle_range $ANGLE_RANGE
    #python3 analysis/statistics.py --result_dir data/"$data_name"
done
