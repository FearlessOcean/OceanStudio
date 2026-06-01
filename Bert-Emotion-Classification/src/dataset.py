from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import config

#1，定义dataset
class ReviewAnalyzeDataset(Dataset):
    def __init__(self,path):
        self.data=pd.read_json(path,lines=True,orient="records").to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor=torch.tensor(self.data[index]['review'],dtype=torch.long)
        target_tensor=torch.tensor(self.data[index]['label'],dtype=torch.float32)
        return input_tensor,target_tensor
    
#2 定义获取dataloader的方法
def get_dataloader(train=True):
    path="..\\data\\processed\\train.json" if train else "..\\data\\processed\\test.json"
    dataset=ReviewAnalyzeDataset(path)

    return DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True)

# train_dataloader=get_dataloader(train=True)
# test_dataloader=get_dataloader(train=False)
# print(len(train_dataloader))
# print(len(test_dataloader))

# for input_tensor,target_tensor in train_dataloader:
#     print(input_tensor.shape)
#     print(target_tensor.shape)
#     break


