# chromaDB使用.py
import chromadb  #  电脑--- vs c++   vip 问答疑
from chromadb.config import Settings   #  对向量数据进行配置（配置名称），调用添加文档的方法，检索的方法
import json
from openai import OpenAI
import os
API_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=API_BASE_URL,
)

def get_embeddings(texts, model="text-embedding-v3"):
    #  texts 是一个包含要获取嵌入表示的文本的列表，
    #  model 则是用来指定要使用的模型的名称
    #  生成文本的嵌入表示。结果存储在data中。 data 数组
    data = client.embeddings.create(input=texts, model=model).data
    # print(data)
    # 返回了一个包含所有嵌入表示的列表
    return [x.embedding for x in data]

#打开train_zh.json文件，并读取数据
with open('./train_zh.json', 'r', encoding='utf-8') as f:
    #  f 是一个文件对象，用于读取文件
    data = [json.loads(line) for line in f]

#将读取的数据按字段拆出放入instructions和outputs中
#在这个业务中，instruction因为需要被查询，需要向量化，output不必要
# 因为数量太大有446189条，只取前10条数据
print(len(data))
instructions = [entry['instruction'] for entry in data[0:10]]
outputs = [entry['output'] for entry in data[0:10]]
print("instructions：",instructions)
print("outputs：",outputs)
# 进行向量化  ，保存到向量数据库中，利用数据库中的向量进行检索

# 负责和向量数据库打交道，接收文档转为向量，并保存到向量数据库中，然后根据需要从向量库中检索出最相似的记录
class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        #当前配置中，数据保存在内存中，如果需要持久化到磁盘，需使用 PersistentClient创建客户端
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # chroma_client = chromadb.PersistentClient(
        #     path="./chroma_data"
        # )

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, instructions, outputs):
        '''向 collection 中添加文档与向量'''
        # get_embeddings(instructions)
        print(len(instructions))
        embeddings = self.embedding_fn(instructions)
        print(len(embeddings))
        self.collection.add(
            embeddings=embeddings,  # 每个文档的向量
            documents=outputs,  # 文档的原文
            ids=[f"id{i}" for i in range(len(outputs))]  # 每个文档的 id
        )

        # print(self.collection.count())

    def search(self, query, top_k):
        '''检索向量数据库
        query是用户的查询，
        top_n指查出top_k个相似高的记录'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            # 最相关的个结果
            n_results=top_k
        )
        return results

# 创建一个向量数据库对象
vector_db = MyVectorDBConnector("demo", get_embeddings)

# 向向量数据库中添加文档
vector_db.add_documents(instructions, outputs)

# user_query = "白癜风"
user_query = "得了白癜风怎么办？"
# 调用 search 方法检索向量数据库
results = vector_db.search(user_query, 3)
print(results)
# 输出结果
for para in results['documents'][0]:
    print(para + "\n")