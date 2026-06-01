import config
from langchain_chroma import Chroma



class VectorStore(object):
    def __init__(self, embedding):
        # 嵌入模型的传入
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        # 返回向量检索其，方便加入chian
        return self.vector_store.as_retriever(search_kwargs = {"k":config.similarity_threshold})

if __name__ == "__main__":
    from langchain_ollama.embeddings import OllamaEmbeddings
    retriever = VectorStore(
        embedding=OllamaEmbeddings(
            model=config.embedding_model_name
        )
    ).get_retriever()
    res = retriever.invoke(
        "我的体重180斤，尺码推荐"
    )
    print(res)