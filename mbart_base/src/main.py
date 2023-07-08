import argparse

import torch
import yaml
from data_preprocess.MSMO_DataModel import SummaryDataModule
from data_preprocess.ch2en_DataModel import SummaryDataModule1
from data_preprocess.en2zh_DataModel import SummaryDataModule3
from data_preprocess.vi2zh_DataModel import SummaryDataModule2
from models.dd_peft_MBart import dd_peft_MBart
from models.peft_MBart import peft_MBart
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
import tracemalloc
tracemalloc.start()

if __name__ == '__main__':
    # prepare for parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="running config path")
    args = parser.parse_args()
    args = yaml.load(open(args.config, "r", encoding="utf-8").read(), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)

    #torch.set_float32_matmul_precision(precision='medium')
    # random seed
    seed_everything(args.random_seed)

    # set logger
    logger = pl_loggers.TensorBoardLogger(args.train_params['default_root_dir']+'vi2zh')#args.model_name_or_path)

    # save checkpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_R2',
                                          save_last=True,
                                          save_top_k=2,
                                          mode='max', )

    trainer = Trainer(**args.train_params,
                      fast_dev_run=False,
                      logger=logger,
                      callbacks=[checkpoint_callback])

    # make dataloader & model
    if args.continue_train is False:
        if args.checkpoint != 'None':
            my_dict = vars(args).copy()
            checkpoint = args.checkpoint
            del my_dict['checkpoint']
            if args.is_ft_peft and args.mode == 'train':
                my_dict['is_lora'] = False
            if args.ch2en_mode == 'dd_summary':
                model = dd_peft_MBart.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **my_dict)
            else:
                model = peft_MBart.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **my_dict)
            # model = dd_peft_MBart(args)
            # state_dict = torch.load(checkpoint)["state_dict"]
            # model.load_state_dict(state_dict, strict=False)
            if args.is_ft_peft:
                model.add_peft(args)
            print('成功加载检查点')
        else:
            if args.ch2en_mode == 'dd_summary':
                model = dd_peft_MBart(args)
            else:
                model = peft_MBart(args)
            print('成功初始化模型')
    if args.src_lang == 'vi_VN':
        summary_data = SummaryDataModule2(args)
    elif args.src_lang == 'zh_CN':
        summary_data = SummaryDataModule1(args)
    elif args.src_lang == 'en_XX':
        summary_data = SummaryDataModule3(args)
    if args.continue_train:
        trainer = Trainer()
        trainer.fit(model=model, ckpt_path=args.checkpoint, datamodule=summary_data)
    else:
        if args.mode == 'train':
            trainer.fit(model=model, datamodule=summary_data)
            trainer.test(model=model, dataloaders=summary_data.test_loader)
        if args.mode == 'test':
            trainer.test(model=model, dataloaders=summary_data.test_loader)
