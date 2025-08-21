import os
import warnings

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv("./.env")

# Initialize OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise Exception("No OpenAI key found. Please set it as an environment variable or in .env")


# Function to generate queries using OpenAI's QWEN
def generate_queries_qwen(original_query):
    """
    基于LLM生成多个query

    Args
        original_query: 用于提出的query

    Returns:
        LLM生成的多个queries
    """
    # 创建OpenAI客户端
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=api_key
    )

    # 创建ChatCompletion
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates multiple search queries based on a single input query."
            },
            {
                "role": "user",
                "content": f"Generate multiple search queries related to: {original_query}"
            },
            {
                "role": "user",
                "content": "OUTPUT (4 queries):"
            }
        ]
    )

    generated_queries = response.choices[0].message.content.strip().split("\n")
    return generated_queries


# 切分文档
def split_documents(all_documents):
    """
    文档分割

    Args
        all_documents: 文档列表

    Returns:
        按照chunk_size和chunk_overlap切割后的文档块
    """
    # Convert the dictionary to a list of Document objects
    docs = [Document(page_content=content, metadata={"doc_id": doc_id}) for doc_id, content in all_documents.items()]

    # create the text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

    # Split the documents
    splits = text_splitter.split_documents(docs)

    return splits


# 相似性搜索
def vector_search_embedding(queries, all_documents):
    """
    基于向量数据库执行向量搜索

    Args:
        queries: 多个查询列表
        all_documents: 所有文档列表

    Returns:
        Dict: 以query作为key，id,page_content,score(距离分数)作为value的查询结果
    """
    # 使用Splitter切分文档
    docs = split_documents(all_documents)

    # 创建嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 创建向量数据库
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

    # 对每个查询执行搜索
    results_dict = {}
    for query in queries:
        # 将query和docs转成向量存储到向量数据库中，再使用similarity_search，这里使用的是距离分数
        results = vectorstore.similarity_search_with_score(query, k=1)

        # 格式化返回结果
        formatted = []
        for doc, score in results:
            formatted.append({
                "id": doc.metadata.get("doc_id", None),
                "page_content": doc.page_content,
                "score": float(score)
            })

        results_dict[query] = formatted

    # 返回最终的结果
    return results_dict


# RRF算法
def reciprocal_rank_fusion(search_results, k=60):
    """
    Reciprocal Rank Fusion算法

    Args:
        search_results: 字典格式，键为查询，值为包含{'id': doc_id, 'page_content': content, 'score': score}的列表
        k: RRF参数，默认60

    Returns:
        list: 重新排序后的文档列表，格式与输入相同
    """
    fused_scores = {}
    print("Initial individual search results rank:")

    for query, doc_list in search_results.items():
        print(f"For query: [{query}]: {[(doc['id'], doc['score'], doc['page_content']) for doc in doc_list]}")

    for query, doc_list in search_results.items():
        # 按照原始score排序，（注意：这里是相似度分数越低越相似，所以升序排列）
        sorted_docs = sorted(doc_list, key=lambda x: x['score'])

        for rank, doc in enumerate(sorted_docs):
            doc_id = doc['id']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    'fused_score': 0,
                    'page_content': doc['page_content'],
                }
            previous_score = fused_scores[doc_id]['fused_score']
            fused_scores[doc_id]['fused_score'] += 1 / (rank + k)
            print(
                f"Updating score for {doc_id} from {previous_score} to {fused_scores[doc_id]['fused_score']} based on rank {rank} in query [{query}]")

    # 按融合分数重新排序并转换为目标格式
    ranked_results = []
    # RRF分数越大越好
    for doc_id, doc_info in sorted(fused_scores.items(), key=lambda x: x[1]['fused_score'], reverse=True):
        reranked_results.append({
            'id': doc_id,
            'page_content': doc_info['page_content'],
            'score': doc_info['fused_score']
        })

    print("Final reranked results:", [(doc['id'], doc['score']) for doc in ranked_results])
    return reranked_results


def generate_output(reranked_results, queries):
    """
    基于重新排序的结果生成输出

    Args:
        reranked_results: 重新排序后的文档列表，格式为[{'id': doc_id, 'page_content': content, 'score': score}]
        queries: 查询列表

    Returns:
        str: 生成的输出文本
    """
    doc_ids = [doc['id'] for doc in reranked_results]
    doc_contents = [doc['page_content'][:100] + '...' if len(doc['page_content']) > 100
                    else doc['page_content'] for doc in reranked_results]

    output = f"Final output based on queries {queries}\n"
    output += f"Reranked documents: {doc_ids}\n"

    return output


# Predefined set of documents (usually these would be from your search database)
all_documents = {
    "doc1": "Climate change and economic impact.",
    "doc2": "Public health concerns due to climate change.",
    "doc3": "Climate change: A social perspective.",
    "doc4": "Technological solutions to climate change.",
    "doc5": "Policy changes needed to combat climate change.",
    "doc6": "Climate change and its impact on biodiversity.",
    "doc7": "Climate change: The science and models.",
    "doc8": "Global warming: A subset of climate change.",
    "doc9": "How climate change affects daily weather.",
    "doc10": "The history of climate change activism."
}

if __name__ == '__main__':
    original_query = "impact of climate change"
    generate_queries = generate_queries_qwen(original_query)

    search_results = vector_search_embedding(generate_queries, all_documents)

    reranked_results = reciprocal_rank_fusion(search_results)

    final_output = generate_output(reranked_results, generate_queries)

    print(final_output)
