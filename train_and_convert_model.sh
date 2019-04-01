TRAIN_DATA=train_bigger_130000.mat
VALID_DATA=valid_bigger_1000.mat
RESULT_FOLDER=big_130000
python3 trainer/train.py --train_data $TRAIN_DATA --val_data $VALID_DATA --result $RESULT_FOLDER
python3 trainer/convert_for_milp.py --name $RESULT_FOLDER
