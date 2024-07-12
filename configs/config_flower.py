device = 'cuda'

num_ep = 100
gen_optimizer_config = dict(
    type='adamw',
    config = dict(
        lr = 3e-4,
        betas=(0.5, 0.999),
    )
)

dis_optimizer_config = dict(
    type='adamw',
    config = dict(
        lr = 3e-4,
        betas=(0.5, 0.999),
    )
)
lr_sche_config = dict(
    type = 'constant',
    config = dict(
        # warm_up_epoch=1
    )
)



####---- model ----####
img_size = 64
dataset_type = 'flower'
model_config = dict(
    dis_config = dict(
        channels=[64, 128, 256, 256, 512]
    ),
    gen_config = dict(
        channels_noise=100, 
        channels=[512, 256, 256, 128, 64]
    )
)

AD_model_config = dict(
    dis_config = dict(
        channels_img=3,
        features_d=64
    ),
    gen_config = dict(
        channels_noise=100,
        channels_img=3,
        features_g=64
    )
)
####---- model ----####



####---- data ----####
data_root = '/home/dmt/shao-tao-working-dir/DATA/OpenDataLab___Oxford_102_Flower/raw/jpg'
train_data_config = dict(
    transform_config = dict(
        img_size = img_size,
        mean=(0.5, ),
        std=(0.5, )
    ),
    dataset_config = dict(
        root_dir=data_root,
        file_lst_txt='train.txt'
    ), 
    data_loader_config = dict(
        batch_size = 16,
        num_workers = 4,
    )
)
val_data_config = dict(
    transform_config = dict(
        img_size = img_size,
        mean=(0.5, ),
        std=(0.5, )
    ),
    dataset_config = dict(
        root_dir=data_root,
        file_lst_txt='val.txt'
    ), 
    data_loader_config = dict(
        batch_size = 16,
        num_workers = 4,
    )
)
####---- data ----####


resume_ckpt_path = None
load_weight_from = None

# ckp
ckp_config = dict(
   save_last=True, 
   every_n_epochs=None,
#    monitor='val_mae',
#    mode='min',
#    filename='{epoch}-{val_mae:.3f}'
)

# trainer config
trainer_config = dict(
    log_every_n_steps=5,
    precision='32',
    # val_check_interval=1, # val after k training batch 0.0-1.0, or a int
    check_val_every_n_epoch=1,
    num_sanity_val_steps=2
)


# LOGGING
enable_wandb = True
wandb_config = dict(
    project = 'gan',
    offline = True
)
ckp_root = f'[{wandb_config["project"]}]'