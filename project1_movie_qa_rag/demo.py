import os
import pandas as pd
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter, 
    LLMChainExtractor,
    DocumentCompressorPipeline
)
from langchain_community.llms import huggingface_pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate 
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 环境变量配置
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ------------------------------
# 辅助函数：分块策略
# ------------------------------
def get_dynamic_splitter(text_length):
    """根据文本长度返回适配的分块器"""
    max_embed_chunk = 1000  # all-mpnet-base-v2有效窗口≈1000字符
    
    if text_length < 800:
        return RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    elif text_length < 3000:
        return RecursiveCharacterTextSplitter(
            chunk_size=max_embed_chunk,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=max_embed_chunk,
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

# ------------------------------
# 辅助函数：数据加载与分块（“数据输入→清洗→结构化分块→统计输出” ）
# ------------------------------
def load_and_split_with_metadata(csv_path):
    """加载CSV数据并生成带元数据的分块"""
    # 验证文件存在性
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_cols = ["Title", "Plot", "Release Year", "Origin/Ethnicity"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV文件必须包含以下列: {required_cols}")
    
    # 数据清洗
    df["Title"] = df["Title"].fillna("未知标题").astype(str)
    df["Plot"] = df["Plot"].fillna("无剧情信息").astype(str)
    df["Release Year"] = df["Release Year"].fillna("未知年份").astype(str)
    df["Origin/Ethnicity"] = df["Origin/Ethnicity"].fillna("未知产地").astype(str)
    
    all_chunks = []
    total_original_length = 0
    longcount = 0  # 统计超长剧情数量
    
    for idx, row in df.iterrows():
        plot_text = row["Plot"]
        text_length = len(plot_text)
        total_original_length += text_length
        
        if text_length > 10000:
            longcount += 1
        
        # 元数据提取
        metadata = {
            "title": row["Title"],
            "year": row["Release Year"],
            "origin": row["Origin/Ethnicity"],
            "movie_id": idx,
            "original_length": text_length,
            "is_long": text_length > 10000
        }
        
        # 动态分块
        splitter = get_dynamic_splitter(text_length)
        plot_chunks = splitter.split_text(plot_text)
        
        # 过滤短分块
        filtered_chunks = [chunk for chunk in plot_chunks if len(chunk) >= 150]
        
        # 添加带元数据的文档
        for chunk in filtered_chunks:
            all_chunks.append(Document(page_content=chunk, metadata=metadata))
    
    # 统计信息
    avg_original_length = total_original_length / len(df) if len(df) > 0 else 0
    print(f"数据处理完成：")
    print(f"- 原始电影数量：{len(df)}部")
    print(f"- 平均剧情长度：{avg_original_length:.0f}字符")
    print(f"- 超长剧情（>10000字符): {longcount}部")
    print(f"- 优化后分块总数：{len(all_chunks)}")
    
    return all_chunks, df

# ------------------------------
# 辅助函数：初始化嵌入模型
# ------------------------------
def initialize_embeddings():
    """初始化嵌入模型（独立模块）"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'auto'},
        encode_kwargs={'normalize_embeddings': True}
    )

# ------------------------------
# 辅助函数：加载或创建向量存储
# ------------------------------
def load_or_create_vectorstore(chunks, embeddings, vector_store_path):
    """加载已有向量存储或创建新的（独立模块）"""
    if os.path.exists(vector_store_path):
        print(f"加载已存在的向量存储: {vector_store_path}")
        return FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("创建新的向量存储...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(vector_store_path, exist_ok=True)
        vectorstore.save_local(vector_store_path)
        return vectorstore

# ------------------------------
# 辅助函数：初始化LLM模型
# ------------------------------
def initialize_llm(model_name="mistral-7b-instruct-v0.1"):
    """初始化生成模型（独立模块）"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"  # 自动分配设备
    )
    
    # 处理pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 构建生成管道
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True,
        do_sample=True
    )
    
    return huggingface_pipeline(pipeline=llm_pipeline)

# ------------------------------
# 辅助函数：构建压缩检索器
# ------------------------------
def build_compression_retriever(vectorstore, embeddings, llm):
    """构建增强型压缩检索器（独立模块）"""
    # 1. 嵌入过滤（低相似度分块）
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.7
    )
    
    # 2. LLM压缩（提取核心信息）
    llm_compressor = LLMChainExtractor.from_llm(llm)
    
    # 3. 压缩管道（先过滤再二次分块再压缩）
    compressor_pipeline = DocumentCompressorPipeline(
        transformers=[
            embeddings_filter,
            CharacterTextSplitter(chunk_size=800, chunk_overlap=50),
            llm_compressor
        ]
    )
    
    # 构建检索器
    return ContextualCompressionRetriever(
        base_compressor=compressor_pipeline,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
    )

# ------------------------------
# 主函数：创建RAG系统
# ------------------------------
def create_rag_system(csv_path, vector_store_path):
    """创建完整的RAG问答系统（主流程）"""
    # 1. 初始化嵌入模型
    embeddings = initialize_embeddings()
    
    # 2. 加载数据并分块
    chunks_with_metadata, original_df = load_and_split_with_metadata(csv_path)
    if not chunks_with_metadata:
        raise ValueError("没有有效的剧情分块，请检查CSV数据")
    
    # 3. 加载或创建向量存储
    vectorstore = load_or_create_vectorstore(
        chunks_with_metadata, 
        embeddings, 
        vector_store_path
    )
    
    # 4. 初始化生成模型
    llm = initialize_llm()
    
    # 5. 构建压缩检索器
    compression_retriever = build_compression_retriever(
        vectorstore, 
        embeddings, 
        llm
    )
    
    # 6. 创建QA链（带自定义提示词）
    prompt_template = """
    你是电影剧情专家，需要基于以下剧情片段和关联的电影信息回答问题。
    请结合电影的标题、年份和产地信息，确保回答准确且有针对性。
    
    参考资料：
    {context}
    
    问题：{question}
    
    回答：
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    qa_chain = create_retrieval_chain(retriever=compression_retriever,  
                             combine_docs_chain=combine_docs_chain
                                             )

    return qa_chain, original_df

# ------------------------------
# 交互主函数
# ------------------------------
def main():
    csv_path = "wiki_movie_plots_deduped.csv"
    vector_store_path = "./movie_vector_store"
    
    try:
        qa_chain, original_df = create_rag_system(csv_path, vector_store_path)
        print("\n电影剧情RAG问答系统（输入'quit'退出）")
        
        while True:
            query = input("\n请输入你的问题: ")
            if query.lower() == 'quit':
                print("谢谢使用！")
                break
            
            result = qa_chain.invoke({"input": query})
            print("\n回答:")
            print(result["answer"])
            
            # 显示关联电影信息
            print("\n关联的电影信息:")
            unique_movies = set()
            for doc in result["context"]:
                movie_id = doc.metadata["movie_id"]
                if movie_id not in unique_movies:
                    unique_movies.add(movie_id)
                    movie_info = original_df.iloc[movie_id]
                    print(f"\n标题: {movie_info['Title']}")
                    print(f"年份: {movie_info['Release Year']}")
                    print(f"产地: {movie_info['Origin/Ethnicity']}")
                    print(f"剧情片段长度: {len(doc.page_content)}字符")
                
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
