noise_mode=uniform
noise_scale=0.01

num_threads=5
exp_name="$noise_mode"_"$noise_scale"
#python parallel_verify.py --offset_range 3 --angle_range 3 --grid_size 0.4 --num_threads $num_threads

# Compute bound by verify
#data_name=verify_offset_"$OFFSET_RANGE"_angle_"$ANGLE_RANGE"_grid_"$grid_size"_thread_"$num_threads"
data_name=verify_offset_3_angle_3_grid_0.4_thread_5none0.05
#for thread_number in $(seq 0 $(expr $num_threads - 1))
#for thread_number in 1
#do
#    /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia verify_thread.jl $exp_name $data_name $thread_number &
#    sleep .5
#done
#wait

offset_err_thresh=3.0
angle_err_thresh=3.0

#/data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia thread_collect_adaptive.jl $data_name $num_threads $offset_err_thresh $angle_err_thresh

# offset_min_size, angle_min_size: don't divide anymore once size smaller than these
level=1
next_folder=$data_name/"$level"
OFFSET_MIN_SIZE=0.06
ANGLE_MIN_SIZE=0.06
#python parallel_verify_adaptive.py --read_from_folder $data_name --write_to_folder $next_folder --num_threads $num_threads --offset_min_size $OFFSET_MIN_SIZE --angle_min_size $ANGLE_MIN_SIZE

# Set timeout=5
time_limit=5.0
for thread_number in 1
do
    /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia verify_thread.jl $exp_name $next_folder $thread_number $time_limit
    sleep .5
done
wait
