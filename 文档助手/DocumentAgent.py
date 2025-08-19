import streamlit as st

# 设置streamlit页面标题和布局
st.set_page_config(page_title="Rag Agent", layout="wide")
# 设置应用的标题
st.title("Rag Agent")

# 上传txt文件，允许上传多个文件
upload_files = st.sidebar.file_uploader(
    label="上传txt文件", type=["txt"], accept_multiple_files=True
)

# 如果没有上传文件，提示用户上传文件
if not upload_files:
    st.info("请先上传TXT文档")
    st.stop()

import os
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import tempfile


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "YOUR LANGSMITH_API_KEY"

# 实现检索器
@st.cache_resource(ttl="1h")
def configure_retriever(upload_files):
    # 读取上传的文档，并写入一个临时目录，用于缓存
    docs = []
    temp_dir = tempfile.TemporaryDirectory(dir=r"D:\\")
    for file in upload_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = TextLoader(temp_filepath, encoding="utf-8")
        docs.extend(loader.load())
    # 文档分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_documents = text_splitter.split_documents(docs)

    # 定义嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = Chroma.from_documents(split_documents, embeddings)

    retriever = vectordb.as_retriever()
    return retriever


# 配置检索器
retriever = configure_retriever(upload_files)

# 如果session_state中没有消息记录或用户点击了清空聊天记录按钮，则初始化聊天记录
if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["messages"] = [{"role": "assistant", "content": "您好，我是AI助手，我可以查询文档"}]

# 加载历史聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 创建检索工具
from langchain.tools.retriever import create_retriever_tool

# 创建用户文档检索的工具
tool = create_retriever_tool(
    retriever,
    "文档检索",
    "用户检索用户提出的问题，并基于检索到的文档内容进行回复"
)
tools = [tool]

# 创建聊天消息历史记录
msgs = StreamlitChatMessageHistory()
# 创建对话缓存区内存
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

# 指令模板
instructions = """您是一个设计用于查询文档来回答问题的代理。
您可以使用文档检索工具，并基于检索内容来回答问题
您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案
如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案"""

# 基础prompt template
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
{chat_history}

New input: {input}
{agent_scratchpad}
"""

# 创建基础提示词模板
base_prompt = PromptTemplate.from_template(base_prompt_template)
# 创建部分填充的提示模板
prompt = base_prompt.partial(instructions=instructions)

# 加载配置文件
load_dotenv("../environment/.env")

# 创建大模型
model = ChatOpenAI(
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    api_key=os.getenv("OPENAI_API_KEY")
)

# 创建react agent
agent = create_react_agent(model, tools, prompt)

# 创建agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

# 创建聊天输入框
user_query = st.chat_input(placeholder="请开始提问吧")

# 如果有用户输入的查询
if user_query:
    # 添加用户消息到session_state
    st.session_state.messages.append({"role": "user", "content": user_query})
    # 显示用户消息
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # 创建streamlit回调处理器
        st_cb = StreamlitCallbackHandler(st.container())
        # agent执行过程中日志回调显示在Streamlit Container
        config = {"callbacks": [st_cb]}
        # 执行agent并获取响应
        response = agent_executor.invoke({"input": user_query}, config=config)
        # 添加助手消息到session_state
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        # 显示助手响应
        st.write(response["output"])
