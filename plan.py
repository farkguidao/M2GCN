import torch
import pytorch_lightning as pl
import yaml
from dataloader.dataloader_with_negtive_sampler import NSDataloader
from models.m2gcn import M2GCNModel
# 用来在晚上连续跑实验的工具
def get_trainer_model_dataloader_from_dir(settings):
    dl = NSDataloader(**settings['data'])
    model = M2GCNModel(N=dl.num_nodes, adj_list=dl.adj_list, **settings['model'])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**settings['callback'])
    trainer = pl.Trainer(callbacks=[checkpoint_callback], **settings['train'])
    return trainer,model,dl

def plan(base_settings,model_replace_key,model_replace_values):
    '''
    :param base_settings: 基础配置
    :param model_replace_key: 取代的超参
    :param model_replace_values: 超参值的列表
    :return:
    '''
    for v in model_replace_values:
        base_settings['model'][model_replace_key] = v
        print('--------------------------------------------------')
        print(model_replace_key, '=', v, 'has bean done!')
        trainer,model,dl=get_trainer_model_dataloader_from_dir(base_settings)
        trainer.fit(model,dl)
        print(model_replace_key, '=', v, 'has finished! result in',trainer.log_dir)
        print('--------------------------------------------------')
        del trainer
        del model
        del dl
    print('finish plan!')

if __name__ == '__main__':
    yaml_path = 'setting/settings.yaml'
    key = 'K'
    values = [6,6,6]
    # key = 'lam'
    # values = [1.,0.5,0.3,0.05,0.01,0.001]
    with open(yaml_path) as f:
        settings = dict(yaml.load(f,yaml.FullLoader))
    plan(settings,key,values)
