
# 文件上传
import streamlit as st
from langchain.callbacks.streamlit import StreamlitCallbackHandler

upload_files = st.sidebar.file_uploader(
    label="上传txt文件", type=["txt"], accept_multiple_files=True
)
if not upload_files:
    st.info("请先上传txt文档")
    st.stop()

# 检索器
import streamlit as st
import tempfile
import os
from langchain.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# 实现检索器
@st.cache_resource(ttl="1h")
def configure_retriever(upload_files):
    # 读取上传的文档，并写入一个临时目录
    docs = []
    temp_dir = tempfile.TemporaryDirectory(dir=r"D:\\")
    for file in upload_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        # 这两行代码就是用 TextLoader 把 UTF-8 编码的文本文件读取成 LangChain 的 Document 对
        # 象列表，并添加到 docs 里，为后续的分词、嵌入和检索做准备
        loader = TextLoader(temp_filepath, encoding="utf-8")
        docs.extend(loader.load())

    # 进行文档的分割
    '''
    chunk_size=300:
        表示每个文本块的最大长度（单位通常是字符数，因为 RecursiveCharacterTextSplitter 是按字符切）。
        假设原文有 1200 个字符：
        第 1 块 → 前 300 个字符
        第 2 块 → 接下来的 300 个字符（考虑 overlap 后会调整）
        依此类推。
        设置的意义：
        太大：每块内容太多，嵌入向量可能不精确（尤其是模型的 token 限制会被拉满）。
        太小：上下文被切碎，语义可能丢失。
    chunk_overlap=30:
        表示相邻两个块之间的重叠字符数。
        这个重叠区域能帮助保持上下文连贯性，避免句子在切割处断裂后丢失关键信息。
        例如：
        第 1 块：字符 0 ~ 299
        第 2 块：从字符 270 开始到 569（因为 300 - 30 = 270）。
        这样，前一段的最后 30 个字符会和后一段的前 30 个字符是相同的。
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = text_splitter.split_documents(docs)

    # 这里使用all-MiniLM-L6-v2模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = Chroma.from_documents(splits, embeddings)

    retriever = vectordb.as_retriever()
    return retriever


# 创建检索工具
from langchain.tools.retriever import create_retriever_tool

tool = create_retriever_tool(
    retriever=configure_retriever(upload_files=upload_files),
    name="文档检索",
    description="用于检索用户提出的问题，并基于检索到的文档内容进行回复。"
)

tools = [tool]

# ReactAgent
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

instructions = """您是一个设计用于查询文档来回答问题的代理。
您可以使用文档检索工具，并基于检索内容来回答问题
您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案
如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案"""

base_prompt_template = """
{instructions}

TOOLS:
___
You have access to the following tools:
{tools}

To use a tool, please use the following format:

.**
Thought: Do I need to use a Tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

.**
when you have a response to say to the human, or if you do not need to use a tool, 
you MUST use the format:

.**
Thought: Do I need to use a tool? No
Final Answer: [your response here]

.**
Begin!

Previous conversation history:
{history}

New input: {input}
{agent_scratchpad}
"""
base_prompt = PromptTemplate.from_template(base_prompt_template)

prompt = base_prompt.partial(instructions=instructions)

# 创建LLM
llm = ChatOpenAI(base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0.7,
                 api_key="sk-4b656b4bbd6e4f8789ebd7cea632c09f")

# 创建ReactAgent
agent = create_react_agent(llm, tools, prompt)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 创建聊天输入框
user_query = st.chat_input(placeholder="您好，有什么我能够帮助你的吗？")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

with st.chat_message("assistant"):
    st_cb = StreamlitCallbackHandler(st.container())
    config = {"callbacks": [st_cb]}
    response = agent_executor.invoke({"input": user_query}, config=config)
    st.session_state.messages.append({"role": "assistant", "content": response["output"]})
    st.write(response["output"])
