import torch

from dataloader.dataloader_with_negtive_sampler import NSDataloader
from models.m2gcn import M2GCNModel
import pytorch_lightning as pl
import yaml
import argparse

def get_trainer_model_dataloader_from_yaml(yaml_path):
    with open(yaml_path) as f:
        settings = dict(yaml.load(f,yaml.FullLoader))

    dl = NSDataloader(**settings['data'])
    model = M2GCNModel(N=dl.num_nodes, adj_list=dl.adj_list, **settings['model'])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**settings['callback'])
    trainer = pl.Trainer(callbacks=[checkpoint_callback], **settings['train'])
    return trainer,model,dl


def train(parser):
    # dl=NSDataloader(batch_size=512*32)
    # model = M2GCNModel(N=dl.num_nodes,adj_list=dl.adj_list,lam=0.5)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='auc',mode='max')
    # trainer = pl.Trainer(max_epochs=10,callbacks=[checkpoint_callback],gpus=1,reload_dataloaders_every_n_epochs=1)
    # trainer.fit(model,dl)
    args = parser.parse_args()
    setting_path = args.setting_path
    trainer,model,dl = get_trainer_model_dataloader_from_yaml(setting_path)
    trainer.fit(model,dl)
def test(parser):
    parser.add_argument('--ckpt_path',type=str,help='model checkpoint path')
    args = parser.parse_args()
    setting_path = args.setting_path
    trainer, model, dl = get_trainer_model_dataloader_from_yaml(setting_path)
    # 加载参数
    state_dict=torch.load(args.ckpt_path)['state_dict']
    model.load_state_dict(state_dict)
    trainer.test(model,dl.test_dataloader())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting_path',type=str,default='setting/settings.yaml')
    parser.add_argument("--test", action='store_true', help='test or train')
    temp_args, _ = parser.parse_known_args()
    if temp_args.test:
        test(parser)
    else:
        train(parser)