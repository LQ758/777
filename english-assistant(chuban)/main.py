from src.core.data_processing import load_sentences_and_paths,get_random_sentence
from src.core.发音评分模块 import record_audio, score_pronunciation
from src.core.语法检查 import analyze_grammar
from src.core.自定义练习模块 import load_custom_data, get_random_custom_sentence
from src.core.处理txt文档 import shuijizhongwen
from src.core.语音转写 import record_audio1,transcribe_audio
import os

def run_speech_scoring():
    """功能1：语音评分（英文句子）"""
    tsv_file = os.path.join("data", "common_voice", "validated.tsv")
    data_records = load_sentences_and_paths(tsv_file)
    random_record = get_random_sentence(data_records)
    reference_text = random_record["sentence"]

    print(f"\n🎯 请朗读以下英文句子：\n{reference_text}")
    input("按下回车键开始录音...")
    audio_data = record_audio()
    score = score_pronunciation(audio_data, reference_text)
    print(f"\n✅ 发音评分: {score:.1f}/100")

def run_grammar_check():
    """功能二：随机输出一句中文，用户自主翻译成英语并阅读，用Whisper进行录音后直接用LanguageTool进行语法检测然后生成错误报告"""

    # 1. 随机获取中文句子
    tsv_file = os.path.join("data", "常用英语口语.txt")
    chinese_sentence = shuijizhongwen(tsv_file)
    if not chinese_sentence:
        return

    print(f"\n🎯 请将以下中文翻译为英文并朗读: \n{chinese_sentence}")
    translated_text = input("请输入英文翻译: \n")

    # 2. 录音并转写
    print(f"\n🔊 请朗读以下英文: \n{translated_text}")
    input("按下回车键开始录音...")
    record_audio1()
    transcribed_text = transcribe_audio()
    print(f"\n📝 录音转写结果: {transcribed_text}")

    # 3. 语法检测
    analysis_result = analyze_grammar(transcribed_text)

    # 4. 结果展示
    if analysis_result.get("status") == "success":
        print("\n✅ 英文语法正确!")
    else:
        print(f"\n❌ 检测到 {analysis_result['error_count']} 处语法问题:")
        for i, error in enumerate(analysis_result['errors'], 1):
            print(f"\n【错误 {i}】")
            print(f"📌 规则ID: {error['rule_id']}")
            print(f"❗ 问题描述: {error['message']}")
            print(f"🔍 上下文: {error['context']}")
            print(f"💡 建议替换: {', '.join(error['replacements'])}")

    # 5. 清理临时文件
    if os.path.exists("temp_recording.wav"):
        os.remove("temp_recording.wav")

def run_custom_exercise():
    """功能3：自定义练习（支持语音评分和语法检测）"""
    file_path = input("\n请输入自定义练习文件路径（.txt 或 .csv）：\n")
    data_records = load_custom_data(file_path)
    if not data_records:
        print("⚠️ 文件内容为空或格式错误！")
        return

    # 用户选择练习模式
    print("\n请选择练习模式：")
    print("1. 语音评分（需提供标准音频）")
    print("2. 语法检测（中文文本）")
    choice = input("请输入选项（1/2）：")

    if choice == "1":
        # 语音评分模式
        random_record = get_random_custom_sentence(data_records)
        reference_text = random_record.get("sentence", "")  # 假设文件中包含参考文本
        if not reference_text:
            print("⚠️ 文件中未找到参考文本！")
            return

        print(f"\n🎯 请朗读以下句子：\n{reference_text}")
        input("按下回车键开始录音...")
        audio_data = record_audio()
        score = score_pronunciation(audio_data, reference_text)
        print(f"\n✅ 发音评分: {score:.1f}/100")

    elif choice == "2":
        # 语法检测模式
        random_record = get_random_custom_sentence(data_records)
        chinese_sentence = random_record.get("chinese", "")  # 假设文件中包含中文句子
        if not chinese_sentence:
            print("⚠️ 文件中未找到中文句子！")
            return

        print(f"\n🎯 请将以下中文翻译为英文并朗读: \n{chinese_sentence}")
        translated_text = input("请输入英文翻译: \n")

        # 录音并转写
        print(f"\n🔊 请朗读以下英文: \n{translated_text}")
        input("按下回车键开始录音...")
        record_audio1()
        transcribed_text = transcribe_audio()
        print(f"\n📝 录音转写结果: {transcribed_text}")

        # 语法检测
        analysis_result = analyze_grammar(transcribed_text)

        # 结果展示
        if analysis_result.get("status") == "success":
            print("\n✅ 英文语法正确!")
        else:
            print(f"\n❌ 检测到 {analysis_result['error_count']} 处语法问题:")
            for i, error in enumerate(analysis_result['errors'], 1):
                print(f"\n【错误 {i}】")
                print(f"📌 规则ID: {error['rule_id']}")
                print(f"❗ 问题描述: {error['message']}")
                print(f"🔍 上下文: {error['context']}")
                print(f"💡 建议替换: {', '.join(error['replacements'])}")

        # 清理临时文件
        if os.path.exists("temp_recording.wav"):
            os.remove("temp_recording.wav")

    else:
        print("❌ 无效选项！")

def main():
    """主程序循环"""
    while True:
        print("\n=== 主菜单 ===")
        print("1. 语音评分（英文句子）")
        print("2. （中译英）语法检测 ")
        print("3. 自定义练习")
        print("4. 退出")
        choice = input("请选择功能（1/2/3/4）：")

        if choice == "1":
            run_speech_scoring()
        elif choice == "2":
            run_grammar_check()
        elif choice == "3":
            run_custom_exercise()
        elif choice == "4":
            print("程序已退出。")
            break
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()