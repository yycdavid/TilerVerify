## Settings for this experiment
DISTANCE_MIN=30
DISTANCE_MAX=60
ANGLE_RANGE=45
num_threads=21

#DISTANCE_MIN=30
#DISTANCE_MAX=32
#ANGLE_RANGE=2
#num_threads=6

INITIAL_GRID_SIZE=2.0
# offset_min_size, angle_min_size: don't divide anymore once size smaller than these
ANGLE_MIN_SIZE=0.4

# Trained model folder
trained_model=trained_models/lidar_quant_061_small/last.pth

noise_mode=gaussian
noise_scale=0.001

num_threads_per_class=$(expr $num_threads / 3)
save_folder=bnn_adaptive

## Generate initial bounding boxes
#python parallel_verify_lidar.py --distance_min $DISTANCE_MIN --distance_max $DISTANCE_MAX --angle_range $ANGLE_RANGE --grid_size $INITIAL_GRID_SIZE --num_threads $num_threads --noise $noise_mode --noise_scale $noise_scale

## Initial verify
data_name=lidar_distance_min_"$DISTANCE_MIN"_max_"$DISTANCE_MAX"_angle_"$ANGLE_RANGE"_grid_"$INITIAL_GRID_SIZE"_thread_"$num_threads""$noise_mode""$noise_scale"_small
for shape in 0 1 2
do
    for thread_number in $(seq 0 $(expr $num_threads_per_class - 1))
    do
        python -m eevbnn eval_bin -e 0.3 --timeout 10 $trained_model --check-cvt --lidar --shape $shape --thread $thread_number --lidar-data-folder $data_name --save-folder $save_folder &
        sleep .5
    done
done
wait

python collect_result_lidar.py --exp_dir $data_name --save-folder $save_folder --num_threads $num_threads_per_class --adaptive


## Loop to adaptively verify
level=0
while true
do
    if [ $level -eq 0 ]
    then
        read_from_folder=$data_name/"$save_folder"
    else
        read_from_folder=$data_name/"$save_folder"/"$level"
    fi

    # See if there is still boxes to solve
    have_to_solve=false
    for shape in 0 1 2
    do
        if [ -f data/$read_from_folder/"$shape"_to_solve.csv ]
        then
            have_to_solve=true
        fi
    done

    # If so, generate divided boxes
    if [ "$have_to_solve" = true ]
    then
        level=$(expr $level + 1)
        next_folder=$data_name/"$save_folder"/"$level"
        python parallel_verify_lidar_adaptive.py --read_from_folder $read_from_folder --write_to_folder $next_folder --num_threads $num_threads --angle_min_size $ANGLE_MIN_SIZE --noise $noise_mode --noise_scale $noise_scale
    else
        echo "Break 2"
        break
    fi

    # See if there is divided boxes generated
    if [ -d data/$next_folder ]
    then
        for shape in 0 1 2
        do
            for thread_number in $(seq 0 $(expr $num_threads_per_class - 1))
            do
                python -m eevbnn eval_bin -e 0.3 --timeout 10 $trained_model --check-cvt --lidar --shape $shape --thread $thread_number --lidar-data-folder $next_folder --save-folder "" &
                sleep .5
            done
        done
        wait

        python collect_result_lidar.py --exp_dir $next_folder --save-folder "" --num_threads $num_threads_per_class --adaptive

    else
        echo "Break 1"
        break
    fi
done
