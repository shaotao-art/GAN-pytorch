device_idx=0
config_p=./configs/config_flower.py
run_name=flower

CUDA_VISIBLE_DEVICES=$device_idx  python run.py \
                        --config $config_p \
                        --run_name $run_name