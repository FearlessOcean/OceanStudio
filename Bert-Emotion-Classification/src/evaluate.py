import torch
import config
from model import ReviewAnalyzeModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer
# 评估
#准备模型、数据集、设备


def evaluate(model,test_dataloader,device):
    total_count = 0
    corrent_count = 0
    for inputs,targets in test_dataloader:
        inputs=inputs.to(device)
        batch_result = predict_batch(model,inputs)
        targets=targets.tolist()  #batch

        for result,target in zip(batch_result,targets):
            result = 1 if result >0.5 else 0
            if result == target:
                corrent_count += 1
            total_count += 1
    return corrent_count / total_count




def run_evaluate():
    print('开始评估')
    #加载资源
    # 1.设备
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 2.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print('---词表加载成功---')

    # 3.模型
    model = ReviewAnalyzeModel(cocab_size=tokenizer.vocab_size,padding_index=tokenizer.pad_token_index).to(device)
    #加载模型参数
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print('---模型加载成功---')

    #4.数据集

    test_dataloader=get_dataloader(train=False)

    #5评估逻辑
    acc=evaluate(model,test_dataloader,device)

    print('评估结果')
    print(f'Acc:{acc}')


if __name__ == '__main__':
    run_evaluate()