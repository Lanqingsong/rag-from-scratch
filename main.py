import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='gbk', errors='replace')
    sys.stderr.reconfigure(encoding='gbk', errors='replace')

from config import Config
from knowledge_base import KnowledgeBase
from llm_client import LLMClient

def main():
    print("=" * 60)
    print("   LangChain + DeepSeek + 本地知识库问答系统")
    print("=" * 60)
    
    try:
        Config.validate()
    except ValueError as e:
        print(f"配置错误: {e}")
        return
    
    kb = KnowledgeBase()
    llm = LLMClient()
    
    print("\n系统初始化中...")
    print("输入 'rebuild' 重新构建向量数据库")
    print("输入 'quit' 或 'exit' 退出程序")
    print("-" * 60)
    
    if not kb.load_vector_store():
        print("未找到本地向量数据库，开始构建...")
        kb.build_vector_store()
    
    print("\n系统已就绪！")
    print("-" * 60)
    
    while True:
        query = input("\n请输入您的问题：")
        
        if query.lower() in ["quit", "exit", "退出"]:
            print("感谢使用，再见！")
            break
        
        if query.lower() in ["rebuild", "重建", "重新构建"]:
            print("开始重新构建向量数据库...")
            kb.build_vector_store()
            print("向量数据库重建完成！")
            continue
        
        if not query.strip():
            print("请输入有效的问题")
            continue
        
        context = kb.search(query)
        
        if context:
            print(f"\n找到 {len(context)} 条相关知识：")
            for i, doc in enumerate(context, 1):
                source = doc.metadata.get("source", "unknown")
                print(f"\n【参考资料 {i}】")
                print(f"来源: {source}")
                print(f"内容:\n{doc.page_content}")
                print("-" * 40)
        
        print("\n【AI回答】")
        if context:
            answer = llm.generate_with_context(query, context)
        else:
            answer = llm.generate_response(query)
        
        print(answer)
        print("\n" + "-" * 60)

if __name__ == "__main__":
    main()