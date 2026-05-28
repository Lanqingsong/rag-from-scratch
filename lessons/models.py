# \RAG\models.py
#可用模型列表，以及获得访问模型的客户端
#实际使用时可以根据自己的实际情况调整
ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max-latest"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"  #
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_3 = "qwen3-235b-a22b"  
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest" 
ALI_TONGYI_EMBEDDING_MODEL = "text-embedding-v3"
ALI_TONGYI_RERANK_MODEL = "gte-rerank-v2"  

DEEPSEEK_API_KEY_OS_VAR_NAME = "Deepseek_Key"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"

#云帆老师的TENCENT_HUNYUAN免费额度最近到期时间：2026-03-24
TENCENT_HUNYUAN_API_KEY_OS_VAR_NAME = "HUNYUAN_API_KEY"
TENCENT_HUNYUAN_URL = "https://api.hunyuan.cloud.tencent.com/v1"
TENCENT_HUNYUAN_TURBO_MODEL = "hunyuan-turbos-latest"
TENCENT_HUNYUAN_REASONER_MODEL = "hunyuan-t1-latest"
TENCENT_HUNYUAN_LONGCONTEXT_MODEL = "hunyuan-large-longcontext"
TENCENT_HUNYUAN_EMBEDDING_MODEL = "hunyuan-embedding"
TENCENT_SECRET_ID_OS_VAR_NAME = "Tencent_SecretId"
TENCENT_SECRET_KEY_OS_VAR_NAME = "Tencent_SecretKey"


BAICHUAN_API_KEY_OS_VAR_NAME = "Baichuan_API_Key"
BAICHUAN_EMBEDDING_MODEL = "Baichuan-Text-Embedding"
BAICHUAN_EMBEDDING_URL = "https://api.baichuan-ai.com/v1/embeddings"

import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
import inspect
from langchain_community.embeddings import BaichuanTextEmbeddings, DashScopeEmbeddings, HunyuanEmbeddings


def get_lc_model_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME), base_url=ALI_TONGYI_URL
                        , model=ALI_TONGYI_MAX_MODEL, temperature=0.7, verbose=False, debug=False):
    """
        通过LangChain获得指定平台和模型的客户端，设定的默认平台和模型为阿里百炼qwen-max-latest
        也可以通过传入api_key，base_url，model三个参数来覆盖默认值
        verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    """
    function_name = inspect.currentframe().f_code.co_name
    if (verbose):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature}")
    if (debug):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature},key：{api_key}")
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=temperature,
                      extra_body={"enable_thinking": False})


def get_ali_model_client(model=ALI_TONGYI_DEEPSEEK_V3, temperature=0.7, verbose=False, debug=False):
    """通过LangChain使用阿里大模型DEEPSEEK_V3"""
    return get_lc_model_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME), base_url=ALI_TONGYI_URL
                               , model=model, temperature=temperature, verbose=verbose, debug=debug)


def get_tencent_model_client(model=TENCENT_HUNYUAN_TURBO_MODEL, temperature=0.7, verbose=False, debug=False):
    """通过LangChain使用腾讯大模型"""
    return get_lc_model_client(api_key=os.getenv(TENCENT_HUNYUAN_API_KEY_OS_VAR_NAME), base_url=TENCENT_HUNYUAN_URL
                               , model=model, temperature=temperature, verbose=verbose, debug=debug)


def get_ds_model_client(model=DEEPSEEK_CHAT_MODEL, temperature=0.7, verbose=False, debug=False):
    """通过LangChain使用DeepSeek大模型"""
    return get_lc_model_client(api_key=os.getenv(DEEPSEEK_API_KEY_OS_VAR_NAME), base_url=DEEPSEEK_URL
                               , model=model, temperature=temperature, verbose=verbose, debug=debug)


def get_baichuan_embeddings():
    """通过LangChain获得一个百川嵌入模型的实例，百川嵌入模型服务限流严重，大概有10~20%的概率访问报错"""
    return BaichuanTextEmbeddings(
        api_key=os.getenv(BAICHUAN_API_KEY_OS_VAR_NAME)
    )


def get_ali_embeddings():
    """通过LangChain获得一个阿里通义千问嵌入模型的实例"""
    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING_MODEL, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    )


def get_tencent_embeddings():
    """通过LangChain获得一个腾讯嵌入模型的实例
    但是注意：需要在腾讯云上额外申请secret_id和secret_key才能使用"""
    return HunyuanEmbeddings(
        hunyuan_secret_id=os.getenv(TENCENT_SECRET_ID_OS_VAR_NAME),
        hunyuan_secret_key=os.getenv(TENCENT_SECRET_KEY_OS_VAR_NAME),
        region="ap-guangzhou",
    )


def get_normal_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME), base_url=ALI_TONGYI_URL,
                      verbose=False, debug=False):
    """
    使用原生api获得指定平台的客户端，但未指定具体模型，缺省平台为阿里云百炼
    也可以通过传入api_key，base_url两个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    """
    function_name = inspect.currentframe().f_code.co_name
    if (verbose):
        print(f"{function_name}-平台：{base_url}")
    if (debug):
        print(f"{function_name}-平台：{base_url},key：{api_key}")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_tencent_normal_client(verbose=False, debug=False):
    """
      使用原生api获得腾讯平台的客户端，但未指定具体模型
      verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    """
    return get_normal_client(api_key=os.getenv(TENCENT_HUNYUAN_API_KEY_OS_VAR_NAME),
                             base_url=TENCENT_HUNYUAN_URL, verbose=verbose, debug=debug)
