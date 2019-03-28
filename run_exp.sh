#python3 trainer/train.py --train_data train_20_10000.mat --val_data valid_20_500.mat --result 20_10000

#python3 trainer/error_estimate.py --exp_name 20_10000 --data test_error_est_10_1.mat

#python3 analysis/heatmap.py --exp_name big_100000/offset_1_angle_1_grid_size_0.1 --type bound

#python3 analysis/statistics.py --exp_name 20_10000/offset_20_angle_20_grid_size_1.0

export JULIA_NUM_THREADS=10

RANGE=1
for grid_size in 0.1
do
    # Generate dataset for estimate and bound
    #python3 main.py --range $RANGE --grid_size $grid_size
    # Compute bound by verify
    /raid/yicheny/software/julia-9d11f62bcb/bin/julia verify.jl $RANGE $grid_size
    # Compute estimate
    #python3 trainer/error_estimate.py --exp_name big_100000 --data test_error_est_"$RANGE"_"$grid_size".mat --grid_size $grid_size
    # Compute statistics and save
    #python3 analysis/statistics.py --exp_name big_100000/offset_"$RANGE"_angle_"$RANGE"_grid_size_"$grid_size"
done

#python3 parallel_verify.py --offset_range 1 --angle_range 1 --grid_size 0.1 --num_threads 30
