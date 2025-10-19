import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模型缓存
_whisper_model = None

def get_whisper_model(model_name="small"):
    """获取Whisper模型（带缓存）"""
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"加载Whisper模型: {model_name}")
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model

def record_audio1(duration=5, fs=16000):
    """录音并返回音频数据"""
    print("\n▶ 开始录音... 请保持安静")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # 等待录音完成
    write("temp_recording.wav", fs, audio)  # 保存临时文件
    return audio, fs

def transcribe_audio(audio_path="temp_recording.wav"):
    """使用Whisper进行语音转写"""
    try:
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("音频文件为空")
        
        logger.info(f"开始转写音频文件: {audio_path} (大小: {file_size} 字节)")
        
        # 获取模型并转写
        model = get_whisper_model()
        result = model.transcribe(audio_path)
        
        transcribed_text = result["text"].strip()
        logger.info(f"转写结果: {transcribed_text}")
        
        if not transcribed_text:
            logger.warning("转写结果为空")
            return ""
        
        return transcribed_text
        
    except Exception as e:
        logger.error(f"Whisper转写失败: {str(e)}")
        raise e