OFFSET_RANGE=40
ANGLE_RANGE=60
NUM_THREADS=20

OFFSET_ERR_THRESH=2.65
ANGLE_ERR_THRESH=3.69

if false
then
for grid_size in 0.05 0.2 0.4 0.8
do
    data_name=verify_offset_"$OFFSET_RANGE"_angle_"$ANGLE_RANGE"_grid_"$grid_size"_thread_"$NUM_THREADS"none0.05

    python analysis/stats_road.py --result_dir data/"$data_name" --offset_err_thresh $OFFSET_ERR_THRESH --angle_err_thresh $ANGLE_ERR_THRESH
done
fi

INITIAL_GRID_SIZE=0.2
data_name=verify_offset_"$OFFSET_RANGE"_angle_"$ANGLE_RANGE"_grid_"$INITIAL_GRID_SIZE"_thread_"$NUM_THREADS"none0.05thresh3.69
python analysis/stats_road.py --result_dir data/"$data_name" --adaptive --plot_final_size $INITIAL_GRID_SIZE
