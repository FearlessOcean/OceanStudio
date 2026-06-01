import torch
from model import ReviewAnalyzeModel
import config
from tokenizer import JiebaTokenizer


def predict_batch(model,inputs):
    model.eval()
    with torch.no_grad():
        output=model(inputs)

    batch_result = torch.sigmoid(output)

    return batch_result.tolist()




def predict(text,tokenizer,model,device):

    indexes = tokenizer.encode(text,config.SEQ_LEN)
    input_tensor=torch.tensor([indexes],dtype=torch.long)
    input_tensor=input_tensor.to(device)

    batch_result=predict_batch(model,input_tensor)

    return batch_result[0]

def run_predict():
    #加载资源
    # 1.设备
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 2.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print('---词表加载成功---')

    # 3.模型
    model = ReviewAnalyzeModel(tokenizer.vocab_size,tokenizer.pad_token_index).to(device)
    #加载模型参数
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print('---模型加载成功---')




    print("欢迎使用情感分析模型，输入quit或者q退出。")
    while True:
        user_inlut = input(">")
        if user_inlut in ['q','quit']:
            break
        if user_inlut.strip()=='':
            print("请输入内容")
            continue


        result = predict(user_inlut,tokenizer,model,device)
        if result >0.5:
            print(f'GOOD 置信度：{result}')
        else :
            print(f'BED 置信度：{1-result}')




if __name__ == '__main__':
    run_predict()