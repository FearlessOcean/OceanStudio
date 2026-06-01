from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

import config
def process():
    print("start data process")

    dataset = load_dataset('csv',data_files=str(config.RAW_DATASETS_DIR / "online_shopping_10_cats.csv"))['train']
    #过滤数据
    dataset = dataset.remove_columns('cat')
    dataset = dataset.filter(lambda x: x['review'] is not None)

    dataset = dataset.cast_column('label',ClassLabel(names=['0','1']))

    print(dataset.features)
    #划分数据集
    dataset_dict = dataset.train_test_split(
        test_size=0.2,
        stratify_by_column='label')

    # 创建Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'models--google-bert--bert-base-chinese')
    print("Finish")


if __name__ == '__main__':
    process()