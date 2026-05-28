# chunk_size表示块的字符个数，stride为步长，滑动窗口的大小为chunk_size-stride
def sliding_window_chunks(text, count, stride):
    return [text[i:i + count] for i in range(0, len(text), count - stride)]


text = ("自然语言处理（NLP），作为计算机科学、人工智能与语言学的交融之地，致力于赋予计算机解析和处理人类语言的能力。"
        "在这个领域，机器学习发挥着至关重要的作用。利用多样的算法，机器得以分析、领会乃至创造我们所理解的语言。"
        "从机器翻译到情感分析，从自动摘要到实体识别，NLP的应用已遍布各个领域。随着深度学习技术的飞速进步，"
        "NLP的精确度与效能均实现了巨大飞跃。如今，部分尖端的NLP系统甚至能够处理复杂的语言理解任务，"
        "如问答系统、语音识别和对话系统等。NLP的研究推进不仅优化了人机交流，也对提升机器的自主性和智能水平起到了关键作用。")


chunks = sliding_window_chunks(text, 100, 20)

for i, chunk in enumerate(chunks):
    print(f"块 {i} - 长度{len(chunk)}，内容: {chunk}")