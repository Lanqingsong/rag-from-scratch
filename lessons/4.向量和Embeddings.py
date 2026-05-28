import os

from openai import OpenAI

API_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=API_BASE_URL,
)

def get_embeddings(texts, model="text-embedding-v3"):
    #  texts 是一个包含要获取嵌入表示的文本的列表，
    #  model 则是用来指定要使用的模型的名称
    # client.embeddings.create(input=texts, model=model)  请求体
    #  生成文本的嵌入表示。结果存储在data中。 data 数组
    data = client.embeddings.create(input=texts, model=model).data
    print(data)
    # 返回了一个包含所有嵌入表示的列表
    return [x.embedding for x in data]


test_query = ["我爱你","你好"]

vec = get_embeddings(test_query)
#  "我爱你" 文本嵌入表示的列表。
#print(vec)  # [[]]  ---[]
# # # #  "我爱你" 文本的嵌入表示。[1,2,3,4,5] 下标--索引--[0,1,2,3,4]
#print(vec[0])
# # # # # # #  "我爱你" 文本的嵌入表示的维度。3072
#print(len(vec[0]))
print(len(vec))
