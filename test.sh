noise_mode=uniform
noise_scale=0.01

num_threads=5
exp_name="$noise_mode"_"$noise_scale"
python parallel_verify.py --offset_range 3 --angle_range 3 --grid_size 0.4 --num_threads $num_threads

# Compute bound by verify
#data_name=verify_offset_"$OFFSET_RANGE"_angle_"$ANGLE_RANGE"_grid_"$grid_size"_thread_"$num_threads"
data_name=verify_offset_3_angle_3_grid_0.4_thread_5none0.05
for thread_number in $(seq 0 $(expr $num_threads - 1))
#for thread_number in 1
do
    /data/scratch/yicheny/software/julia-9d11f62bcb/bin/julia verify_thread.jl $exp_name $data_name $thread_number &
    sleep .5
done
wait
#/raid/yicheny/software/julia-9d11f62bcb/bin/julia thread_collect.jl $data_name $num_threads

# Set timeout=5
