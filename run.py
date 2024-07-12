from mmengine import Config

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import os
import math
import argparse

import torch
from torch import nn
import torch.nn.functional as F


from run_utils import get_callbacks, get_time_str, get_opt_lr_sch
from my_datasets.flower_dataset import get_flower_train_data, get_flower_test_data
from models.gan import Generator, Discriminator
from cv_common_utils import show_or_save_batch_img_tensor, print_model_num_params_and_size



class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.config = config
        ########## ================ MODEL ==================== ##############
        self.gen = Generator(**config.model_config.gen_config)
        self.dis = Discriminator(**config.model_config.dis_config)
        ########## ================ MODEL ==================== ##############
        self.mse_loss = nn.BCELoss()
        
        self.fix_noise = torch.randn(16, 
                                    config.model_config.gen_config.channels_noise, 
                                    1, 1).to(config.device)

        

    def training_step(self, batch, batch_idx):
        # vis train data image
        if self.global_step == 0:
            b_s = batch.shape[0]
            vis_img = show_or_save_batch_img_tensor(batch, int(math.sqrt(b_s)), denorm=True, mode='return')
            self.logger.experiment.add_image(tag=f'train_dataset', 
                                img_tensor=vis_img, 
                                global_step=self.global_step,
                                dataformats='HWC',
                                )
        # sample function
        if self.global_step % 500 == 0:
            self.sample()
            

        
        gen_opt, dis_opt = self.optimizers()
        
        real_sample = batch
        b_s = real_sample.shape[0]
        
        gen_opt.zero_grad()
        noise = torch.randn(b_s, 
                            self.config.model_config.gen_config.channels_noise, 
                            1, 1).to(real_sample.device)
        fake_sample = self.gen(noise)
        # generator optimization
        dis_fake = self.dis(fake_sample)
        loss_gen = self.mse_loss(dis_fake, torch.ones_like(dis_fake).to(dis_fake.device))
        self.log_dict({'train/gen_loss': loss_gen})
        self.manual_backward(loss_gen)
        gen_opt.step()
            
        
        # discriminator optimization
        dis_opt.zero_grad()
        dis_fake = self.dis(fake_sample.detach())
        dis_real = self.dis(real_sample)
        loss_dis_real = self.mse_loss(dis_real, torch.ones_like(dis_real).to(dis_real.device))
        loss_dis_fake = self.mse_loss(dis_fake, torch.zeros_like(dis_fake).to(dis_fake.device))
        loss_dis = loss_dis_real + loss_dis_fake
        self.manual_backward(loss_dis)
        dis_opt.step()
        self.log_dict({'train/dis_loss_real': loss_dis_real})
        self.log_dict({'train/dis_loss_fake': loss_dis_fake})
        


            
    @torch.no_grad()
    def sample(self):
        self.gen.eval()
        generated = self.gen(self.fix_noise)
        print(generated.shape)
        vis_img = show_or_save_batch_img_tensor(generated, 4, denorm=True, mode='return')
        print(vis_img.shape)
        self.logger.experiment.add_image(tag=f'generated', 
                            img_tensor=vis_img, 
                            global_step=self.global_step,
                            dataformats='HWC',
                            )
        self.gen.train()
            
            

    def configure_optimizers(self):
        opt_gen = get_opt_lr_sch(self.config.gen_optimizer_config, 
                              self.config.lr_sche_config,  
                              self.gen)
        opt_dis = get_opt_lr_sch(self.config.dis_optimizer_config, 
                        self.config.lr_sche_config,  
                        self.dis)
        return (opt_gen, opt_dis)
    




def run(args):
    config = Config.fromfile(args.config)
    config = modify_config(config, args)
    
    # make ckp accord to time
    time_str = get_time_str()
    config.ckp_root = '-'.join([time_str, config.ckp_root, f'[{args.run_name}]'])
    config.ckp_config['dirpath'] = config.ckp_root
    os.makedirs(config.ckp_root, exist_ok=True)
    config.run_name = args.run_name
    # logger
    
    # wandb_logger = None
    # if config.enable_wandb:
    #     wandb_logger = WandbLogger(**config.wandb_config,
    #                             name=args.wandb_run_name)
    #     wandb_logger.log_hyperparams(config)
    logger = TensorBoardLogger(save_dir=config.ckp_root,
                               name=config.run_name)
    
    # DATA
    print('getting data...')
    if config.dataset_type == 'flower':
        train_data, train_loader = get_flower_train_data(config.train_data_config)
        val_data, val_loader = get_flower_test_data(config.val_data_config)
    print(f'len train_data: {len(train_data)}, len val_loader: {len(train_loader)}.')
    print(f'len val_data: {len(val_data)}, len val_loader: {len(val_loader)}.')
    print('done.')


    
    # MODEL
    print('getting model...')
    model = Model(config)
    print_model_num_params_and_size(model)
    print(model)
    if 'load_weight_from' in config and config.load_weight_from is not None:
        # only load weights
        state_dict = torch.load(config.load_weight_from)['state_dict']
        model.load_state_dict(state_dict)
        print(f'loading weight from {config.load_weight_from}')
    print('done.')
    
    
    callbacks = get_callbacks(config.ckp_config)
    config.dump(os.path.join(config.ckp_root, 'config.py'))
    
    #TRAINING
    print('staring training...')
    resume_ckpt_path = config.resume_ckpt_path if 'resume_ckpt_path' in config else None
    
    if args.find_lr:
        max_steps = args.max_steps
    else:
        max_steps = -1
    trainer = pl.Trainer(accelerator=config.device,
                         max_epochs=config.num_ep,
                         callbacks=callbacks,
                         logger=logger,
                        #  enable_progress_bar=False,
                         max_steps=max_steps,
                        #  gradient_clip_val=1.0,
                         **config.trainer_config
                         )
    
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=resume_ckpt_path,
                )

def get_args():
    parser = argparse.ArgumentParser()
    # required args
    parser.add_argument("--config", required=True, type=str, help="path to mmcv config file")
    parser.add_argument("--run_name", required=True, type=str, help="wandb run name")
    parser.add_argument("--find_lr", action='store_true', help="whether to find learning rate")
    parser.add_argument("--max_steps", type=int, default=-100, help='max step to run when find lr')

    # common args to overwrite config
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--wd', type=float, help='Weight decay')
    
    args = parser.parse_args()
    return args

def modify_config(config, args):
    if args.lr is not None:
        config['optimizer_config']['config']['lr'] = args.lr
    if args.wd is not None:
        config['optimizer_config']['config']['weight_decay'] = args.wd
    return config


    

if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(42)
    run(args)