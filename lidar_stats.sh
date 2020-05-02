DISTANCE_MIN=30
DISTANCE_MAX=60
ANGLE_RANGE=45
num_threads=21

noise_mode=gaussian
noise_scale=0.001

save_folder=bnn

if false
then
for grid_size in 1.0 0.5
do
    data_name=lidar_distance_min_"$DISTANCE_MIN"_max_"$DISTANCE_MAX"_angle_"$ANGLE_RANGE"_grid_"$grid_size"_thread_"$num_threads""$noise_mode""$noise_scale"_small

    python analysis/stats_lidar.py --result_dir data/"$data_name"/"$save_folder"
done
fi

INITIAL_GRID_SIZE=2.0
save_folder=bnn_adaptive

data_name=lidar_distance_min_"$DISTANCE_MIN"_max_"$DISTANCE_MAX"_angle_"$ANGLE_RANGE"_grid_"$INITIAL_GRID_SIZE"_thread_"$num_threads""$noise_mode""$noise_scale"_small

python analysis/stats_lidar.py --result_dir data/"$data_name"/"$save_folder" --adaptive
