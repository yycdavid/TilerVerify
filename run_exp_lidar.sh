DISTANCE_MIN_TRAIN=15
DISTANCE_MIN_VALID=20
DISTANCE_MAX_TRAIN=65
DISTANCE_MAX_VALID=60
ANGLE_RANGE_TRAIN=50
ANGLE_RANGE_VALID=45
noise_mode=gaussian
noise_scale=0.001
python3 generate_data_lidar.py --mode train --distance_max_train $DISTANCE_MAX_TRAIN --distance_min_train $DISTANCE_MIN_TRAIN --angle_range_train $ANGLE_RANGE_TRAIN --distance_max_valid $DISTANCE_MAX_VALID --distance_min_valid $DISTANCE_MIN_VALID --angle_range_valid $ANGLE_RANGE_VALID --noise $noise_mode --noise_scale $noise_scale --target_dir_name lidar_train_"$noise_mode"_"$noise_scale"

#TRAIN_DATA=train_bigger_130000.mat
#VALID_DATA=valid_bigger_1000.mat
#RESULT_FOLDER="$noise_mode"_"$noise_scale"
#python3 trainer/train.py --dataset_folder train_"$noise_mode"_"$noise_scale" --train_data $TRAIN_DATA --val_data $VALID_DATA --result $RESULT_FOLDER
#python3 trainer/convert_for_milp.py --name $RESULT_FOLDER

#OFFSET_RANGE=40
#ANGLE_RANGE=60
#num_threads=20
##exp_name="$noise_mode"_"$noise_scale"
#exp_name=uniform_0.1
#for grid_size in 0.1
#do
#    export JULIA_NUM_THREADS=$num_threads
#    # Generate dataset for estimate and bound
#    #python3 parallel_verify.py --offset_range $OFFSET_RANGE --angle_range $ANGLE_RANGE --grid_size $grid_size --num_threads #$num_threads --noise $noise_mode --noise_scale $noise_scale
#    #python3 parallel_verify.py --offset_range 3 --angle_range 3 --grid_size 0.1 --num_threads $num_threads
#
#    # Compute bound by verify
#    data_name=verify_offset_"$OFFSET_RANGE"_angle_"$ANGLE_RANGE"_grid_"$grid_size"_thread_"$num_threads""$noise_mode""$noise_#scale"
#    #data_name=verify_offset_3_angle_3_grid_0.1_thread_20none0.05
#    #for thread_number in $(seq 0 $(expr $num_threads - 1))
#    #do
#    #    /raid/yicheny/software/julia-9d11f62bcb/bin/julia verify_thread.jl $exp_name $data_name $thread_number &
#    #    sleep .5
#    #done
#    #wait
#    #/raid/yicheny/software/julia-9d11f62bcb/bin/julia thread_collect.jl $data_name $num_threads
#    # Compute estimate
#    #python3 generate_data.py --mode estimate --offset_range $OFFSET_RANGE --angle_range $ANGLE_RANGE --grid_size $grid_size #--target_dir_name $data_name --noise $noise_mode --noise_scale $noise_scale
#    python3 trainer/error_estimate.py --exp_name $exp_name --target_dir_name $data_name --grid_size $grid_size
#    # Get heatmap
#    python3 analysis/heatmap.py --result_dir data/"$data_name" --offset_range $OFFSET_RANGE --angle_range $ANGLE_RANGE
#    python3 analysis/statistics.py --result_dir data/"$data_name"
#done
