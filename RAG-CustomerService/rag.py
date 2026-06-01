import configparser
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from file_history_store import get_history
import config
from vector_stores import VectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt
class RagService(object):
    def __init__(self):
        self.vector_service = VectorStore(
            embedding=config.EMBEDDING_MODEL,

        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知的参考资料为主，简洁和专业的回答用户问题。参考资料：{references}"),
                ("system","并且我提供用户历史纪录如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问{question}")
            ]
        )
        self.chat_model = config.CHAT_MODEL
        self.str_parser = StrOutputParser()
        #下面这个函数用到的组件要定义在 它的上面   parper定义在它下面会报错
        self.chain = self.__get_chain()

    def __get_chain(self):
        # 获取最终执行链
        retriever = self.vector_service.get_retriever()
        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"
            formated_str = ""
            for doc in docs:
                formated_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formated_str

        def format_for_retriever(value):
            return value["question"]
        def format_for_prompt_template(value):
            new_value = {}
            new_value["question"] = value["question"]["question"]
            new_value["references"] = value["references"]
            new_value["history"] = value["question"]["history"]
            return new_value

        # chain = (
        #     {
        #         "question":RunnablePassthrough(),
        #         "references": RunnableLambda(format_for_retriever) | retriever | format_document
        #     } | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt |self.chat_model | self.str_parser
        chain = (
            {
                "question":RunnablePassthrough(),
                "references": RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt |self.chat_model | self.str_parser

        )

        enhance_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="question",
            history_messages_key="history"
        )
        return enhance_chain

if __name__ == "__main__":

    res = RagService().chain.invoke({"question":"针织毛衣如何保养？"},session_config)
    print(res)

#RunnablePassthrough.assign(context=itemgetter("input") | retriever | format_document） 就可以了