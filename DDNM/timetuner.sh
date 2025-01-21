INPUT_ROOT="/tmp2/ICML2025"
EXP="/tmp2/ICML2025/ddnm"
IMAGE_FOLDER="/tmp2/ICML2025/ddnm_timetuner/imagenet"
# python main.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 5 --timetuner val
python main.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 5 --timetuner train
python main.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 10 --timetuner train
python main.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 20 --timetuner train

