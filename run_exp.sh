#python3 trainer/train.py --train_data train_20_10000.mat --val_data valid_20_500.mat --result 20_10000

#python3 trainer/test_error.py --exp_name 20_10000 --data test_error_est_10_1.mat

#python3 analysis/heatmap.py --exp_name 20_10000/offset_20_angle_20_grid_size_1.0 --type bound

python3 analysis/statistics.py --exp_name 20_10000/offset_20_angle_20_grid_size_1.0
