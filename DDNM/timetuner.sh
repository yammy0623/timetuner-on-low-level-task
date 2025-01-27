INPUT_ROOT="/tmp2/ICML2025"
EXP="/tmp2/ICML2025/ddnm"
IMAGE_FOLDER="/tmp2/ICML2025/ddnm_timetuner/imagenet"
export CUDA_VISIBLE_DEVICES=1

# python main.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 5 --timetuner val
python main.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 5 --timetuner train
python main.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 10 --timetuner train
python main.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i $IMAGE_FOLDER --input_root $INPUT_ROOT --step_nums 20 --timetuner train

# 20: [948, 905, 859, 803, 752, 790, 774, 723, 723, 723, 235, 271, 351, 300, 260, 386, 297, 186, 78, 1]
# 10: [905 802 790 752 723 234 237 210 142 1]
# 5: [802 774 0 0 1]  0 是 nan