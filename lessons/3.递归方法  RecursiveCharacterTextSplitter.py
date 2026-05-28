from langchain.text_splitter import RecursiveCharacterTextSplitter

text = ("自然语言处理（NLP），作为计算机科学、人工智能与语言学的交融之地，致力于赋予计算机解析和处理人类语言的能力。"
        "在这个领域，机器学习发挥着至关重要的作用。利用多样的算法，机器得以分析、领会乃至创造我们所理解的语言。"
        "从机器翻译到情感分析，从自动摘要到实体识别，NLP的应用已遍布各个领域。随着深度学习技术的飞速进步，"
        "NLP的精确度与效能均实现了巨大飞跃。如今，部分尖端的NLP系统甚至能够处理复杂的语言理解任务，"
        "如问答系统、语音识别和对话系统等。NLP的研究推进不仅优化了人机交流，也对提升机器的自主性和智能水平起到了关键作用。")
'''
    RecursiveCharacterTextSplitter 是一个用于将文本分割成较小块的工具。核心思想是根据一组分隔符（separators）逐步分割文本，
    直到每个块的大小都符合预设的chunk_size。如果某个块仍然过大，它会继续递归地分割，直到满足条件为止。
    其默认字符列表为 `["\n\n", "\n", " ", ""]`，这种设置首先尝试保持段落、句子和单词的完整性。
    它特别适用于需要递归地按字符拆分文本的场景，例如处理超长文档或嵌套结构的文本
    chunk_size = 分割长度
    chunk_overlap = 重叠长度 
    chunk_overlap 是在文本块尺寸大于 chunk_size 时，为了确保相邻文本块之间有部分重叠才发挥作用的
    若分隔符分割出的文本块尺寸已经小于 chunk_size，RecursiveCharacterTextSplitter 则认为没必要进行重叠处理
    此时相邻文本块间不一定有重叠
'''

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=20,  # 分块  20 -- 不会大于20
#     chunk_overlap=10,
# )

'''更改separators，某些语言如中文、日文和泰语没有明确的词边界。
为了避免默认分隔符列表导致的词语拆分问题，可以覆盖默认分隔符列表
以包括其他标点符号，如句号、逗号以及零宽度空格等'''
splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=10,
    length_function=len,
    separators=["\n\n",".",",","。","，","\n"," ","",],
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"块 {i + 1} - 长度{len(chunk)}，内容: {chunk}")
