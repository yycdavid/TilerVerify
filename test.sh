## Settings for this experiment
OFFSET_RANGE=40
ANGLE_RANGE=60

INITIAL_GRID_SIZE=0.2

NUM_THREADS=20

TIME_LIMIT=5.0

OFFSET_ERR_THRESH=2.65
ANGLE_ERR_THRESH=3.69
#OFFSET_ERR_THRESH=5.0
#ANGLE_ERR_THRESH=5.0

# offset_min_size, angle_min_size: don't divide anymore once size smaller than these
OFFSET_MIN_SIZE=0.06
ANGLE_MIN_SIZE=0.06

# Trained model folder
noise_mode=uniform
noise_scale=0.01
exp_name="$noise_mode"_"$noise_scale"

#export JULIA_NUM_THREADS=$NUM_THREADS

## Generate initial bounding boxes
python parallel_verify.py --offset_range $OFFSET_RANGE --angle_range $ANGLE_RANGE --grid_size $INITIAL_GRID_SIZE --num_threads $NUM_THREADS

## Initial verify
data_name=verify_offset_"$OFFSET_RANGE"_angle_"$ANGLE_RANGE"_grid_"$INITIAL_GRID_SIZE"_thread_"$NUM_THREADS"none0.05
for thread_number in $(seq 0 $(expr $NUM_THREADS - 1))
do
    /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia verify_thread.jl $exp_name $data_name $thread_number $TIME_LIMIT &
    sleep 0.5
done
wait

/data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia thread_collect_adaptive.jl $data_name $NUM_THREADS $OFFSET_ERR_THRESH $ANGLE_ERR_THRESH


## Loop to adaptively verify
level=0
while true
do
    if [ $level -eq 0 ]
    then
        read_from_folder=$data_name
    else
        read_from_folder=$data_name/"$level"
    fi

    if [ -f data/$read_from_folder/to_solve.csv ]
    then
        level=$(expr $level + 1)
        next_folder=$data_name/"$level"
        python parallel_verify_adaptive.py --read_from_folder $read_from_folder --write_to_folder $next_folder --num_threads $NUM_THREADS --offset_min_size $OFFSET_MIN_SIZE --angle_min_size $ANGLE_MIN_SIZE

        if [ -d data/$next_folder ]
        then
            for thread_number in $(seq 0 $(expr $NUM_THREADS - 1))
            do
                /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia verify_thread.jl $exp_name $next_folder $thread_number $TIME_LIMIT &
                sleep 0.5
            done
            wait

            /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia thread_collect_adaptive.jl $next_folder $NUM_THREADS $OFFSET_ERR_THRESH $ANGLE_ERR_THRESH
        else
            echo "Break 1"
            break
        fi
    else
        echo "Break 2"
        break
    fi
done
