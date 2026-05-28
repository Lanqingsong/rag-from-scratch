from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from docx import Document


def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    """
    从 PDF 文件中（按指定页码）提取文字

    思路：
    1. 使用 pdfplumber 逐页解析 PDF
    2. 根据指定页码过滤需要处理的页面
    3. 提取页面中的文本容器（LTTextContainer）内容
    4. 将所有文本按行暂存后，重新合并被换行的段落
    5. 处理英文单词的连字符情况
    6. 按空行分割段落

    参数：
    filename: PDF文件路径
    page_numbers: 指定要提取的页码列表（从0开始），None表示提取全部
    min_line_length: 最小行长度，短于此长度的行将被视为段落分隔符

    返回：
    段落列表（每个元素为一个段落字符串）
    """
    paragraphs = []
    buffer = ''  # 用于暂存正在构建的段落
    full_text = ''  # 存储提取的原始文本

    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):  # 遍历每一页
        # 过滤页码：如果指定了页码范围且当前页不在范围内，则跳过
        if page_numbers is not None and i not in page_numbers:
            continue

        # 遍历页面中的每个元素
        for element in page_layout:
            # 仅处理文本容器（排除图片、表格等元素）
            # 这个判断的作用是：只处理纯文本内容，跳过图片/表格/图形等非文本元素
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'  # 保留换行用于后续处理

    # 按空行分隔，重新组织段落
    lines = full_text.split('\n')
    for text in lines:
        # 处理有效行（长度超过阈值）
        if len(text) >= min_line_length:
            # 处理连字符情况：行以连字符结尾时，拼接时不加空格并去除连字符
            if not text.endswith('-'):
                buffer += ' ' + text  # 普通行拼接
            else:
                buffer += text.strip('-')  # 处理被换行分割的单词
        # 遇到空行时，将暂存内容作为段落存入列表
        elif buffer:
            paragraphs.append(buffer.strip())  # 去除首尾空格后存储
            buffer = ''  # 重置暂存区

    # 处理最后一个段落
    if buffer:
        paragraphs.append(buffer.strip())

    return paragraphs


def extract_text_from_docx(filename, min_line_length=1):
    """
    从 DOCX 文件中提取文字

    思路：
    1. 使用 python-docx 库读取文档
    2. 提取所有段落的原始文本
    3. 按空行分割并重组段落
    4. 过滤短行（假设为标题）
    5. 处理英文连字符情况

    参数：
    filename: DOCX文件路径
    min_line_length: 最小行长度，短于此长度的行将被视为段落分隔符

    返回：
    段落列表（每个元素为一个段落字符串）
    """
    paragraphs = []
    buffer = ''
    full_text = ''

    # 读取文档
    doc = Document(filename)

    # 提取原始文本（保留换行符）
    for para in doc.paragraphs:
        full_text += para.text + '\n'

    # 处理文本内容
    lines = full_text.split('\n')
    for line in lines:
        # 有效行处理（长度超过阈值）
        if len(line) >= min_line_length:
            # 处理连字符情况
            if not line.endswith('-'):
                buffer += ' ' + line
            else:
                buffer += line.strip('-')
        # 遇到分隔行时存储段落
        elif buffer:
            paragraphs.append(buffer.strip())
            buffer = ''

    # 处理最后一个段落
    if buffer:
        paragraphs.append(buffer.strip())

    return paragraphs

# paragraphs = extract_text_from_pdf("人事管理流程.pdf", page_numbers=[1, 3], min_line_length=10)
# print(paragraphs)
docx_filename = "人事管理流程.docx"
# 读取Word文件
paragraphs = extract_text_from_docx(docx_filename, min_line_length=10)
print(paragraphs)
print(len(paragraphs))
print(paragraphs[0])




