import os
import numpy as np
import traceback

# 导入音素级评分模块
try:
    from .音素评分模块 import PhonemeScorer, DetailedPronunciationResult
    from .音素特征提取 import AcousticFeatureExtractor, PhonemeAligner, PronunciationQualityAssessor
    PHONEME_SCORING_AVAILABLE = True
    print('✅ 音素级评分模块加载成功')
except ImportError as e:
    print(f'⚠️ 音素级评分模块导入失败: {e}')
    PHONEME_SCORING_AVAILABLE = False

# 延迟导入，避免启动时的依赖问题
def _import_dependencies():
    """延迟导入依赖库，减少Flask重载触发"""
    try:
        # 设置环境变量减少库的调试输出
        import os
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        os.environ.setdefault('PYTORCH_DISABLE_VERSION_CHECK', '1')
        
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import librosa
        return torch, Wav2Vec2Processor, Wav2Vec2ForCTC, librosa
    except ImportError as e:
        print(f"依赖库导入失败: {e}")
        return None, None, None, None

def record_audio(duration=3, sr=16000):
    """录音函数，返回音频数据"""
    try:
        import sounddevice as sd
        print("🎙️ 录音开始... 按下回车结束（3秒）")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()  # 等待录音完成
        return audio.flatten()
    except ImportError:
        print("警告: sounddevice库未安装，无法录音")
        return np.zeros(sr * duration)

def score_pronunciation(audio_data, reference_text, detailed=False):
    """使用 Wav2Vec2 评估发音准确性
    
    Args:
        audio_data: 音频数据
        reference_text: 参考文本
        detailed: 是否返回音素级详细评分
    
    Returns:
        float或DetailedPronunciationResult: 简单评分或详细评分结果
    """
    try:
        # 延迟导入依赖
        torch, Wav2Vec2Processor, Wav2Vec2ForCTC, librosa = _import_dependencies()
        
        if torch is None:
            raise ImportError("必要的依赖库未安装，请运行: pip install -r requirements.txt")
        
        # 检查音频数据
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("音频数据为空")
        
        print(f"音频数据长度: {len(audio_data)} 采样点")
        
        # 使用相对路径，确保在不同环境下都能找到模型
        model_name = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "models", "wav2vec2-base-960h")
        
        if not os.path.exists(model_name):
            raise FileNotFoundError(f"模型路径不存在: {model_name}")
        
        # 检查模型文件是否完整
        required_files = ['config.json', 'pytorch_model.bin', 'vocab.json']
        for file in required_files:
            file_path = os.path.join(model_name, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"模型文件缺失: {file_path}")
        
        print(f"正在加载模型: {model_name}")
        
        # 设置设备（GPU或CPU）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 加载模型和处理器
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model = model.to(device)
        model.eval()  # 设置为评估模式
        
        print("模型加载成功")

        # 确保音频数据是float32类型
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 音频数据归一化
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        print("音频预处理完成")

        # 将音频数据转为模型输入格式
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
        # 注意：processor返回的是一个字典/BatchFeature，需使用**解包或通过键访问
        inputs = {k: v.to(device) for k, v in dict(inputs).items()}
        
        print("开始语音识别...")
        
        # 推理
        with torch.no_grad():
            # 正确调用方式：使用关键字参数解包，避免属性访问错误
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        print(f"语音识别结果: {transcription}")

        # 评分逻辑（基于 Levenshtein 距离）
        try:
            from Levenshtein import distance
            print("使用Levenshtein距离计算相似度")
        except ImportError:
            try:
                from python_Levenshtein import distance
                print("使用python-Levenshtein距离计算相似度")
            except ImportError:
                # 如果没有Levenshtein库，使用改进的字符串相似度算法
                print("警告: 未找到Levenshtein库，使用改进的相似度计算")
                def improved_similarity(str1, str2):
                    str1, str2 = str1.lower().strip(), str2.lower().strip()
                    if len(str1) == 0 or len(str2) == 0:
                        return 0
                    
                    # 计算字符级别的相似度
                    char_similarity = sum(1 for c in str1 if c in str2) / max(len(str1), len(str2))
                    
                    # 计算单词级别的相似度
                    words1 = set(str1.split())
                    words2 = set(str2.split())
                    if len(words1) == 0 or len(words2) == 0:
                        word_similarity = 0
                    else:
                        word_similarity = len(words1.intersection(words2)) / max(len(words1), len(words2))
                    
                    # 综合评分
                    return (char_similarity * 0.4 + word_similarity * 0.6)
                
                score = 100 * improved_similarity(transcription, reference_text)
                final_score = max(0, min(score, 100))
                print(f"改进相似度评分完成: {final_score}")
                return final_score

        # 使用Levenshtein距离计算
        distance_score = distance(transcription.lower(), reference_text.lower())
        max_distance = max(len(transcription), len(reference_text))
        
        if max_distance == 0:
            similarity = 1.0
            score = 100
        else:
            similarity = 1 - (distance_score / max_distance)
            score = 100 * similarity
        
        final_score = max(0, min(score, 100))
        print(f"Levenshtein距离评分完成: {final_score}")
        print(f"  转录文本: '{transcription}'")
        print(f"  参考文本: '{reference_text}'")
        print(f"  编辑距离: {distance_score}")
        print(f"  最大长度: {max_distance}")
        print(f"  相似度: {similarity:.3f}")
        
        # 如果需要详细评分且音素模块可用，进行音素级分析
        if detailed and PHONEME_SCORING_AVAILABLE:
            try:
                print("开始音素级详细分析...")
                phoneme_scorer = PhonemeScorer()
                detailed_result = phoneme_scorer.analyze_pronunciation_detailed(
                    audio_data, reference_text, model, processor, sr=16000
                )
                # 使用音素级评分作为最终评分
                detailed_result.overall_score = max(detailed_result.overall_score, final_score * 0.8)
                print(f"音素级分析完成，最终评分: {detailed_result.overall_score:.1f}")
                return detailed_result
            except Exception as e:
                print(f"音素级分析失败，回退到简单评分: {e}")
                # 如果音素级分析失败，返回简单评分包装成详细结果
                if detailed:
                    return create_simple_detailed_result(final_score, transcription, reference_text)
        
        # 如果不需要详细评分或音素模块不可用，返回简单评分
        if detailed:
            return create_simple_detailed_result(final_score, transcription, reference_text)
        
        return final_score
        
    except Exception as e:
        print(f"发音评分过程中出错: {str(e)}")
        traceback.print_exc()
        
        # 提供具体的错误信息和建议
        if "CUDA" in str(e):
            error_msg = "GPU内存不足，建议使用CPU模式或减少音频长度"
        elif "model" in str(e).lower():
            error_msg = "模型加载失败，请检查模型文件是否完整"
        elif "audio" in str(e).lower():
            error_msg = "音频处理失败，请检查音频格式和长度"
        else:
            error_msg = f"未知错误: {str(e)}"
        
        print(f"错误详情: {error_msg}")
        raise RuntimeError(f"发音评分失败: {error_msg}")


def create_simple_detailed_result(score: float, transcription: str, reference_text: str):
    """创建简化的详细评分结果"""
    if not PHONEME_SCORING_AVAILABLE:
        # 如果音素模块不可用，返回简单的字典结果
        return {
            'overall_score': score,
            'transcription': transcription,
            'reference_text': reference_text,
            'phoneme_scores': [],
            'pronunciation_issues': [],
            'improvement_suggestions': ['请检查发音节奏和清晰度'],
            'detailed_available': False
        }
    
    # 使用DetailedPronunciationResult类
    from .音素评分模块 import DetailedPronunciationResult
    
    return DetailedPronunciationResult(
        overall_score=score,
        phoneme_scores=[],
        word_scores=[],
        pronunciation_issues=['未进行音素级分析'],
        improvement_suggestions=['建议多练习发音清晰度'],
        duration_analysis={'total_duration': 0},
        pitch_analysis={}
    )


def score_pronunciation_detailed(audio_data, reference_text):
    """返回详细发音评分结果"""
    return score_pronunciation(audio_data, reference_text, detailed=True)