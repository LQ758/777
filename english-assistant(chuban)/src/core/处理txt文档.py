import re
import random
import os

def shuijizhongwen(file_path):
    try:
        # 构建文件的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "..", "data", "常用英语口语.txt")

        # 检查文件存在性
        if not os.path.exists(file_path):
            print(f"错误：文件 '{file_path}' 未找到")
            return None

        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # 正则表达式：使用两步处理法解决后行断言长度问题
        # 第一步：使用更简单的句子分割（可能包含一些错误分割）
        sentences = re.split(r'(?<=[.!?。？！…])\s+', text)

        # 第二步：合并因缩写被错误分割的句子
        fixed_sentences = []
        skip_next = False

        for i in range(len(sentences)):
            if skip_next:
                skip_next = False
                continue

            # 检查是否是常见的缩写结尾
            if i < len(sentences) - 1 and re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|No|vs|etc|e\.g|i\.e|a\.m|p\.m)\.', sentences[i]):
                fixed_sentences.append(sentences[i] + ' ' + sentences[i + 1])
                skip_next = True
            else:
                fixed_sentences.append(sentences[i])

        # 返回随机句子
        if fixed_sentences:
            return random.choice(fixed_sentences)
        else:
            return None

    except Exception as e:
        print(f"错误：{e}")
        return None

# 测试函数
if __name__ == "__main__":
    random_sentence = shuijizhongwen("data/常用英语口语.txt")
    if random_sentence:
        print( random_sentence)