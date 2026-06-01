"""
 基于streamlit 完成 WEB网页上传服务
 通过 streamlit 库的命令去运行python代码得到网页
 streamlit 页面元素发生变法，代码重新执行一遍
"""
import time

import streamlit as st
from knowledge_base import KnowledgeBase

# 添加网页标题
st.title("知识库更新服务")


# st.session_state 字典
# 存在这里保存不会刷新丢失
if 'service' not in st.session_state:
    st.session_state['service'] = KnowledgeBase()
# file_uploader
up_loader_file = st.file_uploader(
    "请上传TXT文件",
    type=["txt"],
    accept_multiple_files=False   #false表示仅接受一个文件的上传
)


if up_loader_file is not None:
    # 提取文件信息
    file_name = up_loader_file.name
    file_type = up_loader_file.type
    file_size = up_loader_file.size /1024  #得到KB
    st.subheader(f"文件名：{file_name}")
    st.write(f"格式:{file_type} | 大小：{file_size:.2f} KB")

    # 获取内容 -> bytes -> decode (utf-8)
    text = up_loader_file.getvalue().decode("utf-8")
    with st.spinner("载入知识库中..."):
        time.sleep(1)
        result = st.session_state['service'].upload_by_str(
            data=text,
            filename=file_name)
        st.write(result)