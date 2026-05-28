import os
import pypdf
from collections import Counter

def analyze_pdf_structure(pdf_path):
    print("=" * 60)
    print(f"PDF结构分析: {os.path.basename(pdf_path)}")
    print("=" * 60)
    
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        num_pages = len(reader.pages)
        
        print(f"\n📄 基本信息:")
        print(f"  总页数: {num_pages}")
        
        all_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            all_text.append(text)
        
        total_chars = sum(len(text) for text in all_text)
        print(f"  总字符数: {total_chars:,}")
        print(f"  平均每页字符数: {total_chars // num_pages:,}")
        
        print(f"\n📊 页面内容分析:")
        for i in range(min(5, num_pages)):
            content = all_text[i].strip()
            char_count = len(content)
            line_count = len(content.split('\n'))
            print(f"  第{i+1}页: {char_count}字符, {line_count}行")
            
            if i == 0:
                print(f"    首页预览: {content[:100]}...")
        
        if num_pages > 5:
            print(f"  ... (还有 {num_pages - 5} 页)")
        
        print(f"\n🔍 文本结构分析:")
        
        full_text = "\n".join(all_text)
        lines = full_text.split('\n')
        print(f"  总行数: {len(lines)}")
        
        line_lengths = [len(line.strip()) for line in lines if line.strip()]
        if line_lengths:
            avg_line_length = sum(line_lengths) / len(line_lengths)
            print(f"  平均行长: {avg_line_length:.1f} 字符")
            print(f"  最短行: {min(line_lengths)} 字符")
            print(f"  最长行: {max(line_lengths)} 字符")
        
        empty_lines = [line for line in lines if not line.strip()]
        print(f"  空行数: {len(empty_lines)} ({len(empty_lines)/len(lines)*100:.1f}%)")
        
        print(f"\n📝 段落结构分析:")
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        print(f"  段落数: {len(paragraphs)}")
        
        if paragraphs:
            para_lengths = [len(p) for p in paragraphs]
            print(f"  平均段落长度: {sum(para_lengths)/len(para_lengths):.1f} 字符")
            print(f"  最短段落: {min(para_lengths)} 字符")
            print(f"  最长段落: {max(para_lengths)} 字符")
            
            long_paragraphs = [p for p in paragraphs if len(p) > 500]
            print(f"  超过500字符的段落: {len(long_paragraphs)}")
            
            print(f"\n  段落长度分布:")
            length_ranges = [
                (0, 100, "0-100"),
                (100, 300, "100-300"),
                (300, 500, "300-500"),
                (500, 800, "500-800"),
                (800, float('inf'), "800+")
            ]
            for min_len, max_len, label in length_ranges:
                count = len([p for p in paragraphs if min_len <= len(p) < max_len])
                print(f"    {label}字符: {count} ({count/len(paragraphs)*100:.1f}%)")
        
        print(f"\n🔗 标题/章节特征分析:")
        
        potential_headers = []
        for line in lines:
            line = line.strip()
            if len(line) > 0 and len(line) < 100:
                if line[0].isdigit() or line.startswith(('第', '一、', '二、', '三、', '1.', '2.', '3.')):
                    potential_headers.append(line)
        
        print(f"  可能的标题行: {len(potential_headers)}")
        if potential_headers:
            print(f"  示例标题: {potential_headers[:5]}")
        
        print(f"\n❓ 问答格式检测:")
        qa_patterns = ['Q:', 'A:', '问：', '答：', '问题：', '答案：', '?']
        qa_count = sum(1 for line in lines if any(pattern in line for pattern in qa_patterns))
        print(f"  包含问答标记的行: {qa_count}")
        if qa_count > 0:
            print(f"  可能是问答格式文档")
        
        print(f"\n📋 文本样本 (前500字符):")
        print(f"  {full_text[:500]}...")
        
        print(f"\n💡 切分建议:")
        
        if qa_count > len(lines) * 0.1:
            print("  建议: 按问答对切分，每个问答作为一个chunk")
            print("  推荐参数: chunk_size=800, chunk_overlap=0")
        elif len(paragraphs) > num_pages * 2:
            print("  建议: 按段落切分，保持段落完整性")
            print("  推荐参数: chunk_size=600, chunk_overlap=50")
        else:
            print("  建议: 按固定字符数切分")
            print("  推荐参数: chunk_size=500, chunk_overlap=50")
    
    print("=" * 60)

if __name__ == "__main__":
    pdf_path = "工业视觉100问第一部分.pdf"
    if os.path.exists(pdf_path):
        analyze_pdf_structure(pdf_path)
    else:
        print(f"PDF文件不存在: {pdf_path}")