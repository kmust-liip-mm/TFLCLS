import pickle
import numpy
import torch
from torch.utils.data import Dataset



def file_reader(file_path):
    f = open(file_path, 'rb')
    dic = pickle.load(f)
    f.close()
    return dic


class MSMODataset(Dataset):
    """Summarization dataset"""
    def __init__(self, args, mode):
        self.args = args
        # get origin data
        if mode == 'train':
            path = args.data_path + 'MSMO_train.pkl'
        if mode == 'val':
            path = args.data_path + 'MSMO_val.pkl'
        if mode == 'test':
            path = args.data_path + 'MSMO_test.pkl'
        self.mode = mode
        self.dic = file_reader(path)
        self.dataList = list(self.dic.keys())


    def __len__(self):
        return len(self.dic)

    def __getitem__(self, idx):
        dataId = self.dataList[idx]
        if self.args.model == 'text_only_bart':
            return {
                "dataId": dataId,
                "summary": self.dic[dataId]["summary"],
                "sentences": self.dic[dataId]["sentences"],
            }
        else:
            if self.args.fusion_type != 2:
                imgs_feature = None
            else:
                imgs_feature = torch.tensor(numpy.load(self.args.preprocee_data_path + self.mode + '/' + dataId + ".npy"))[:self.args.max_image_num]
            return {
                "dataId": dataId,
                "summary": self.dic[dataId]["summary"],
                "sentences": self.dic[dataId]["sentences"],
                "logits_per_text_softmax": torch.from_numpy(self.dic[dataId]["logits_per_text_softmax"])[:,
                                           :self.args.max_image_num],
                "imgs_feature": imgs_feature,
                "pooled_imgs_feature": self.dic[dataId]["pooled_imgs_feature"]
            }

