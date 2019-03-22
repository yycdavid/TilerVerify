#python3 trainer/train.py --train_data train_20_10000.mat --val_data valid_20_500.mat --result 20_10000

#python3 trainer/error_estimate.py --exp_name 20_10000 --data test_error_est_10_1.mat

#python3 analysis/heatmap.py --exp_name 20_10000/offset_20_angle_20_grid_size_1.0 --type bound

#python3 analysis/statistics.py --exp_name 20_10000/offset_20_angle_20_grid_size_1.0

RANGE=10
for grid_size in 0.5 0.25 0.1
do
    # Generate dataset for estimate and bound
    python3 main.py --range $RANGE --grid_size $grid_size
    # Compute bound by verify
    /Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia verify.jl $RANGE $grid_size
    # Compute estimate
    python3 trainer/error_estimate.py --exp_name 20_10000 --data test_error_est_"$RANGE"_"$grid_size".mat --grid_size $grid_size
    # Compute statistics and save
    python3 analysis/statistics.py --exp_name 20_10000/offset_"$RANGE"_angle_"$RANGE"_grid_size_"$grid_size"
done
