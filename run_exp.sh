#python3 trainer/train.py --data data --result test_run

#python3 trainer/test_error.py --exp_name test_run --data test_error_est.mat

python3 analysis/heatmap.py --exp_name test_run/offset_4_angle_2_grid_size_1.0 --type bound
