from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI

# model
ollama_base_url = "http://127.0.0.1:11434/v1"
embedding_model_name = "qwen3-embedding:0.6b"
BASE_URL="https://hiapi.online/v1"
API_KEY="sk-k7fyyns8nfJxqFLYcQJC41rJdBFSBeHjTZuig7lZrOPmEpi5"

EMBEDDING_MODEL = OllamaEmbeddings(
    model=embedding_model_name,
)

CHAT_MODEL= ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model="gemini-2.5-flash"
)



md5_path = "./md5.text"

# chroma_db
collection_name = "RAG"
persist_directory = "./chroma_db" # 数据库本地存储文件夹


# 分割后每一个文本段最大长度
chunk_size = 1000
# 连续文本段之间的字符重叠数量
chunk_overlap = 100
# 自然段落划分分隔符
separators = ["\n\n","\n",",",".","，","。","？","?"," "]
# 字符串分割阈值
max_split_char_number = 1000


# 相似度检索阈值 返回前多少个相似数据
similarity_threshold = 1

session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}