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
TIME_LIMIT=5.0
# offset_min_size, angle_min_size: don't divide anymore once size smaller than these
ANGLE_MIN_SIZE=0.4

# Trained model folder
noise_mode=gaussian
noise_scale=0.001
RESULT_FOLDER=lidar_small_"$noise_mode"_"$noise_scale"

num_threads_per_class=$(expr $num_threads / 3)
exp_name=$RESULT_FOLDER
save_folder=milp_adaptive

## Generate initial bounding boxes
#python parallel_verify_lidar.py --distance_min $DISTANCE_MIN --distance_max $DISTANCE_MAX --angle_range $ANGLE_RANGE --grid_size $INITIAL_GRID_SIZE --num_threads $num_threads --noise $noise_mode --noise_scale $noise_scale

## Initial verify
data_name=lidar_distance_min_"$DISTANCE_MIN"_max_"$DISTANCE_MAX"_angle_"$ANGLE_RANGE"_grid_"$INITIAL_GRID_SIZE"_thread_"$num_threads""$noise_mode""$noise_scale"_small
if false
then
for shape in 0 1 2
do
    for thread_number in $(seq 0 $(expr $num_threads_per_class - 1))
    do
        /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia verify_thread_lidar.jl $exp_name $data_name $shape $thread_number $save_folder $TIME_LIMIT &
        sleep .5
    done
done
wait

/data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia thread_collect_lidar_adaptive.jl $data_name $num_threads_per_class $save_folder
fi

next_folder=$data_name/"$save_folder"/1
/data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia thread_collect_lidar_adaptive.jl $next_folder $num_threads_per_class ""


## Loop to adaptively verify
#level=0
level=1
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
                /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia verify_thread_lidar.jl $exp_name $next_folder $shape $thread_number "" $TIME_LIMIT &
                sleep .5
            done
        done
        wait

        /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia thread_collect_lidar_adaptive.jl $next_folder $num_threads_per_class ""

    else
        echo "Break 1"
        break
    fi
done
