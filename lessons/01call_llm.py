#学会通过LangChain 调用大模型，实现交互
import os

from langchain_openai import ChatOpenAI

from models import get_lc_model_client

"""
原生的聊天交互
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
client = OpenAI(api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
completion = client.chat.completions.create(
    model=TONGYI_MAX_MODEL,
    #model="deepseek-chat",
    messages=[
        {"role": "user", "content": "你好，请问如何做红烧牛肉？"}
    ]
)
"""

# 获取API KEY
# MODEL_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# #拿到一个访问大模型客户端
# client = ChatOpenAI(api_key=MODEL_API_KEY,
#                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#                     model="qwen-max-latest")
'''
注意，因为上面的两行代码在每个程序中都几乎会用到，
所以被抽取出来放到了models.py中，成为了单独函数
比如get_lc_model_client已经设定了缺省大模型，也允许传入参数自行设定模型
在后面的代码中则会直接使用models.py中函数来获得访问大模型的客户端，不再另外说明
'''
client = get_lc_model_client()

#实现一个翻译助手
msg = [
    ('system','请将以下的内容翻译成英文'),
    ('human','你好，你今天过得好吗？'),
]
#invoke-调用/执行
result = client.invoke(msg)
print(result)
