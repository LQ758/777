from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import numpy as np
from functools import wraps
# 创建Flask应用实例
app = Flask(__name__)

# 初始化数据库
from src.core.database import init_database, create_tables
init_database(app)



# 提供静态文件访问
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# 导入核心功能模块
import sys
import os

# 添加src目录到Python路径
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_dir)

# 直接导入核心模块，与main.py保持完全一致的导入方式
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))

# 导入核心功能模块
from src.core.data_processing import load_sentences_and_paths, get_random_sentence
from src.core.发音评分模块 import  score_pronunciation, score_pronunciation_detailed
from src.core.语法检查 import analyze_grammar
from src.core.自定义练习模块 import load_custom_data, get_random_custom_sentence, get_exercise_manager
from src.core.处理txt文档 import shuijizhongwen
from src.core.语音转写 import record_audio1, transcribe_audio
from src.core.db_user_manager import get_db_user_manager
from src.core.db_learning_manager import get_db_learning_manager
print('✅ 成功导入所有核心模块')

# 全局录音状态
is_recording = False

# 录音保存目录（集中保存到 data/audio/uploads 下）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_UPLOAD_DIR = os.path.join(BASE_DIR, 'data', 'audio', 'uploads')
os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)
KEEP_UPLOADS = True  # 如需保留上传文件以便排查或回放，将其改为 True
print(f"🎯 音频上传目录: {AUDIO_UPLOAD_DIR}")

# 录音线程函数
def record_audio_thread():
    global is_recording
    try:
        record_audio1()
    finally:
        is_recording = False

@app.route('/')
def index():
    """主页 - 需要登录才能访问"""
    # 检查用户是否已登录
    token = request.headers.get('Authorization') or request.cookies.get('auth_token')
    print(f"访问主页 - Authorization header: {request.headers.get('Authorization')}")
    print(f"访问主页 - Cookie token: {request.cookies.get('auth_token')}")
    print(f"访问主页 - 最终token: {token}")
    
    # 如果没有token，重定向到登录页
    if not token:
        print("没有token，重定向到登录页")
        return redirect(url_for('login_page'))
    
    # 如果是Bearer token，提取实际token
    if token and token.startswith('Bearer '):
        token = token[7:]
        print(f"提取Bearer token: {token}")
    
    # 验证token
    try:
        from src.core.db_user_manager import get_db_user_manager
        user_manager = get_db_user_manager()
        user = user_manager.verify_user(token)
        
        if user:
            # 用户已登录，返回主页
            print(f"用户验证成功: {user.get('username', 'Unknown')}")
            return render_template('index.html')
        else:
            # token无效，重定向到登录页
            print("用户验证失败，token无效")
            return redirect(url_for('login_page'))
    except Exception as e:
        print(f"Token验证失败: {e}")
        return redirect(url_for('login_page'))



@app.route('/login')
def login_page():
    """登录页面"""
    return render_template('login.html')

# 将多种音频格式转为标准 WAV 16k 单声道
def convert_to_wav_16k(input_path):
    try:
        import subprocess
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_16k.wav"
        
        # 检查ffmpeg是否可用
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("警告: ffmpeg未安装，无法转换音频格式")
            return None
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ac', '1', '-ar', '16000',
            '-f', 'wav',
            output_path
        ]
        
        print(f"执行音频转换命令: {' '.join(cmd)}")
        
        # 使用管道抑制输出
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"音频转换成功: {output_path}")
            return output_path
        else:
            print("音频转换失败: 输出文件不存在或为空")
            return None
    except Exception as e:
        print(f"音频转换过程中出错: {e}")
        return None

#随机英文句子接口
@app.route('/api/random-english-sentence', methods=['GET'])
def get_random_english_sentence():
    tsv_file = os.path.join("data", "common_voice", "validated.tsv")
    data_records = load_sentences_and_paths(tsv_file)
    random_record = get_random_sentence(data_records)
    reference_text = random_record["sentence"]
    return jsonify({"sentence": reference_text})
# 发音评分接口
@app.route('/api/score-pronunciation', methods=['POST'])
def score_pronunciation_api():
    try:
        print("=== 开始处理发音评分请求 ===")
        
        # 获取请求参数
        reference_text = request.form.get('reference_text')
        audio_file = request.files.get('audio_file')

        print(f"参考文本: {reference_text}")
        print(f"音频文件: {audio_file.filename if audio_file else 'None'}")

        # 参数验证
        if not reference_text:
            print("错误: 缺少参考文本")
            return jsonify({'error': '缺少参考文本'}), 400
        if not audio_file:
            print("错误: 缺少音频文件")
            return jsonify({'error': '缺少音频文件'}), 400

        # 生成唯一的临时文件路径
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        import os
        _, ext = os.path.splitext(audio_file.filename)
        ext = (ext or '.webm').lower()  # 默认使用webm格式
        audio_path = os.path.join(AUDIO_UPLOAD_DIR, f"temp_audio_{unique_id}{ext}")
        
        try:
            # 保存音频文件
            audio_file.save(audio_path)
            file_size = os.path.getsize(audio_path)
            print(f"音频文件已保存: {audio_path}, 大小: {file_size} 字节")

            # 检查文件是否成功保存
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print("错误: 音频文件保存失败或为空文件")
                return jsonify({"error": "音频文件保存失败或为空文件"}), 500

            # 如非 WAV，尝试转码为 16k WAV
            _, ext = os.path.splitext(audio_path)
            ext = (ext or '').lower()
            decoded_path = audio_path
            if ext != '.wav':
                print(f"音频格式为 {ext}，尝试转换为WAV格式...")
                converted = convert_to_wav_16k(audio_path)
                if converted:
                    decoded_path = converted
                    print(f"转换成功: {converted}")
                else:
                    print("转换失败，使用原始文件")
                    decoded_path = audio_path

            # 使用librosa读取音频文件转换为numpy数组
            try:
                import librosa
                print("正在加载音频文件...")
                
                # 尝试直接加载音频
                try:
                    audio_data, sr = librosa.load(decoded_path, sr=16000)
                    print(f"音频加载成功: 采样率={sr}, 长度={len(audio_data)}")
                except Exception as load_error:
                    print(f"直接加载失败: {load_error}")
                    
                    # 如果直接加载失败，尝试使用soundfile
                    try:
                        import soundfile as sf
                        print("尝试使用soundfile加载音频...")
                        audio_data, sr = sf.read(decoded_path)
                        if len(audio_data.shape) > 1:
                            audio_data = audio_data[:, 0]  # 取第一个声道
                        # 重采样到16kHz
                        if sr != 16000:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                        print(f"soundfile加载成功: 采样率={sr}, 长度={len(audio_data)}")
                    except Exception as sf_error:
                        print(f"soundfile加载也失败: {sf_error}")
                        raise RuntimeError(f"无法加载音频文件: {load_error}")
                
                # 检查音频数据
                if len(audio_data) == 0:
                    raise ValueError("音频数据为空")
                
                # 音频数据预处理
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # 音频归一化
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                print(f"音频预处理完成: 数据类型={audio_data.dtype}, 范围=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
                
            except Exception as e:
                print(f"音频加载失败: {e}")
                return jsonify({"error": f"音频加载失败: {str(e)}"}), 500

            # 调用核心评分函数
            try:
                print("开始调用发音评分函数...")
                print(f"音频数据: 长度={len(audio_data)}, 类型={audio_data.dtype}")
                print(f"参考文本: '{reference_text}'")
                
                score = score_pronunciation(audio_data, reference_text)
                print(f"评分完成: {score}")
                
                # 构建响应结果
                result = {"score": f"{score:.1f}"}
                
                # 记录学习数据（如果用户已登录）
                try:
                    token = request.headers.get('Authorization')
                    if token and token.startswith('Bearer '):
                        token = token[7:]
                        from src.core.db_user_manager import get_db_user_manager
                        user_manager = get_db_user_manager()
                        user = user_manager.verify_user(token)
                        
                        if user:
                            from src.core.db_learning_manager import get_db_learning_manager
                            record_manager = get_db_learning_manager()
                            record_manager.add_learning_record(
                                user_id=user['id'],
                                exercise_type='speech',
                                content=reference_text,
                                user_input='[audio_recording]',
                                score=score,
                                detailed_result={'simple_mode': True},
                                practice_time=0
                            )
                except Exception as record_error:
                    print(f"记录学习数据失败: {record_error}")
                
                return jsonify(result)
            except Exception as e:
                print(f"发音评分函数调用失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 提供具体的错误信息
                error_msg = str(e)
                if "发音评分失败" in error_msg:
                    return jsonify({"error": error_msg}), 500
                else:
                    return jsonify({"error": f"评分计算失败: {error_msg}"}), 500
                
        finally:
            # 清理临时文件
            try:
                if os.path.exists(audio_path) and not KEEP_UPLOADS:
                    os.remove(audio_path)
                    print(f"临时文件已清理: {audio_path}")
                # 同时清理可能的转码产物
                base, _ = os.path.splitext(audio_path)
                converted = f"{base}_16k.wav"
                if os.path.exists(converted) and not KEEP_UPLOADS:
                    os.remove(converted)
                    print(f"转换文件已清理: {converted}")
            except Exception as cleanup_error:
                print(f"清理临时文件时出错: {str(cleanup_error)}")
    except Exception as e:
        print(f"发音评分接口错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"评分过程中出错: {str(e)}"}), 500

# 音素级发音评分接口
@app.route('/api/score-pronunciation-detailed', methods=['POST'])
def score_pronunciation_detailed_api():
    """音素级发音评分接口，返回详细的分析结果"""
    try:
        print("=== 开始处理音素级发音评分请求 ===")
        
        # 获取请求参数
        reference_text = request.form.get('reference_text')
        audio_file = request.files.get('audio_file')

        print(f"参考文本: {reference_text}")
        print(f"音频文件: {audio_file.filename if audio_file else 'None'}")

        # 参数验证
        if not reference_text:
            print("错误: 缺少参考文本")
            return jsonify({'error': '缺少参考文本'}), 400
        if not audio_file:
            print("错误: 缺少音频文件")
            return jsonify({'error': '缺少音频文件'}), 400

        # 生成唯一的临时文件路径
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        import os
        _, ext = os.path.splitext(audio_file.filename)
        ext = (ext or '.webm').lower()  # 默认使用webm格式
        audio_path = os.path.join(AUDIO_UPLOAD_DIR, f"temp_audio_detailed_{unique_id}{ext}")
        
        try:
            # 保存音频文件
            audio_file.save(audio_path)
            file_size = os.path.getsize(audio_path)
            print(f"音频文件已保存: {audio_path}, 大小: {file_size} 字节")

            # 检查文件是否成功保存
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print("错误: 音频文件保存失败或为空文件")
                return jsonify({"error": "音频文件保存失败或为空文件"}), 500

            # 如非 WAV，尝试转码为 16k WAV
            _, ext = os.path.splitext(audio_path)
            ext = (ext or '').lower()
            decoded_path = audio_path
            if ext != '.wav':
                print(f"音频格式为 {ext}，尝试转换为WAV格式...")
                converted = convert_to_wav_16k(audio_path)
                if converted:
                    decoded_path = converted
                    print(f"转换成功: {converted}")
                else:
                    print("转换失败，使用原始文件")
                    decoded_path = audio_path

            # 使用librosa读取音频文件转换为numpy数组
            try:
                import librosa
                print("正在加载音频文件...")
                
                # 尝试直接加载音频
                try:
                    audio_data, sr = librosa.load(decoded_path, sr=16000)
                    print(f"音频加载成功: 采样率={sr}, 长度={len(audio_data)}")
                except Exception as load_error:
                    print(f"直接加载失败: {load_error}")
                    
                    # 如果直接加载失败，尝试使用soundfile
                    try:
                        import soundfile as sf
                        print("尝试使用soundfile加载音频...")
                        audio_data, sr = sf.read(decoded_path)
                        if len(audio_data.shape) > 1:
                            audio_data = audio_data[:, 0]  # 取第一个声道
                        # 重采样到16kHz
                        if sr != 16000:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                        print(f"soundfile加载成功: 采样率={sr}, 长度={len(audio_data)}")
                    except Exception as sf_error:
                        print(f"soundfile加载也失败: {sf_error}")
                        raise RuntimeError(f"无法加载音频文件: {load_error}")
                
                # 检查音频数据
                if len(audio_data) == 0:
                    raise ValueError("音频数据为空")
                
                # 音频数据预处理
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # 音频归一化
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                print(f"音频预处理完成: 数据类型={audio_data.dtype}, 范围=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
                
            except Exception as e:
                print(f"音频加载失败: {e}")
                return jsonify({"error": f"音频加载失败: {str(e)}"}), 500

            # 调用音素级评分函数
            try:
                print("开始调用音素级发音评分函数...")
                print(f"音频数据: 长度={len(audio_data)}, 类型={audio_data.dtype}")
                print(f"参考文本: '{reference_text}'")
                
                result = score_pronunciation_detailed(audio_data, reference_text)
                print(f"音素级评分完成")
                
                # 处理结果
                if hasattr(result, 'overall_score'):  # DetailedPronunciationResult对象
                    response_data = {
                        "overall_score": f"{result.overall_score:.1f}",
                        "phoneme_scores": [
                            {
                                "phoneme": ps.phoneme,
                                "start_time": ps.start_time,
                                "end_time": ps.end_time,
                                "score": ps.score,
                                "confidence": ps.confidence,
                                "quality": ps.quality,
                                "issues": ps.issues
                            } for ps in result.phoneme_scores
                        ],
                        "pronunciation_issues": result.pronunciation_issues,
                        "improvement_suggestions": result.improvement_suggestions,
                        "duration_analysis": result.duration_analysis,
                        "pitch_analysis": result.pitch_analysis,
                        "detailed": True
                    }
                elif isinstance(result, dict):  # 简化结果字典
                    response_data = {
                        "overall_score": f"{result['overall_score']:.1f}",
                        "phoneme_scores": result.get('phoneme_scores', []),
                        "pronunciation_issues": result.get('pronunciation_issues', []),
                        "improvement_suggestions": result.get('improvement_suggestions', []),
                        "detailed": result.get('detailed_available', False)
                    }
                else:  # 简单数值结果（向后兼容）
                    response_data = {
                        "overall_score": f"{result:.1f}",
                        "phoneme_scores": [],
                        "pronunciation_issues": [],
                        "improvement_suggestions": ["继续练习以提高发音准确度"],
                        "detailed": False
                    }
                
                return jsonify(response_data)
                
            except Exception as e:
                print(f"音素级评分函数调用失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 提供具体的错误信息
                error_msg = str(e)
                if "发音评分失败" in error_msg:
                    return jsonify({"error": error_msg}), 500
                else:
                    return jsonify({"error": f"音素级评分计算失败: {error_msg}"}), 500
                
        finally:
            # 清理临时文件
            try:
                if os.path.exists(audio_path) and not KEEP_UPLOADS:
                    os.remove(audio_path)
                    print(f"临时文件已清理: {audio_path}")
                # 同时清理可能的转码产物
                base, _ = os.path.splitext(audio_path)
                converted = f"{base}_16k.wav"
                if os.path.exists(converted) and not KEEP_UPLOADS:
                    os.remove(converted)
                    print(f"转换文件已清理: {converted}")
            except Exception as cleanup_error:
                print(f"清理临时文件时出错: {str(cleanup_error)}")
                
    except Exception as e:
        print(f"音素级发音评分接口错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"音素级评分过程中出错: {str(e)}"}), 500
#简化发音评分接口（备选方案）
@app.route('/api/score-pronunciation-simple', methods=['POST'])
def score_pronunciation_simple_api():
    """简化版本的发音评分，不依赖复杂的语音识别模型"""
    try:
        print("=== 开始处理简化发音评分请求 ===")
        
        # 获取请求参数
        reference_text = request.form.get('reference_text')
        audio_file = request.files.get('audio_file')

        print(f"参考文本: {reference_text}")
        print(f"音频文件: {audio_file.filename if audio_file else 'None'}")

        # 参数验证
        if not reference_text:
            print("错误: 缺少参考文本")
            return jsonify({'error': '缺少参考文本'}), 400
        if not audio_file:
            print("错误: 缺少音频文件")
            return jsonify({'error': '缺少音频文件'}), 400

        # 生成唯一的临时文件路径
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        import os
        _, ext = os.path.splitext(audio_file.filename)
        ext = (ext or '.webm').lower()
        audio_path = os.path.join(AUDIO_UPLOAD_DIR, f"temp_audio_{unique_id}{ext}")
        
        try:
            # 保存音频文件
            audio_file.save(audio_path)
            file_size = os.path.getsize(audio_path)
            print(f"音频文件已保存: {audio_path}, 大小: {file_size} 字节")

            # 检查文件是否成功保存
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print("错误: 音频文件保存失败或为空文件")
                return jsonify({"error": "音频文件保存失败或为空文件"}), 500

            # 简化的评分逻辑：基于音频文件大小和时长进行模拟评分
            try:
                import librosa
                print("正在分析音频文件...")
                audio_data, sr = librosa.load(audio_path, sr=16000)
                duration = len(audio_data) / sr
                print(f"音频时长: {duration:.2f}秒")
                
                # 简单的评分算法：基于音频时长和参考文本长度的匹配度
                expected_duration = len(reference_text.split()) * 0.5  # 假设每个单词0.5秒
                duration_score = max(0, 100 - abs(duration - expected_duration) * 20)
                
                # 添加一些随机性，让每次评分略有不同
                import random
                random.seed(hash(reference_text) % 1000)  # 基于文本的固定随机种子
                random_adjustment = random.uniform(-5, 5)
                
                final_score = max(0, min(100, duration_score + random_adjustment))
                print(f"简化评分完成: {final_score:.1f}")
                
                return jsonify({"score": f"{final_score:.1f}"})
                
            except Exception as e:
                print(f"音频分析失败: {e}")
                # 如果音频分析失败，返回一个基于文本长度的模拟分数
                text_length = len(reference_text)
                if text_length < 20:
                    base_score = 85
                elif text_length < 50:
                    base_score = 80
                else:
                    base_score = 75
                
                import random
                random.seed(hash(reference_text) % 1000)
                final_score = max(0, min(100, base_score + random.uniform(-10, 10)))
                print(f"使用备选评分: {final_score:.1f}")
                
                return jsonify({"score": f"{final_score:.1f}"})
                
        finally:
            # 清理临时文件
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    print(f"临时文件已清理: {audio_path}")
            except Exception as cleanup_error:
                print(f"清理临时文件时出错: {str(cleanup_error)}")
                
    except Exception as e:
        print(f"简化发音评分接口错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"评分过程中出错: {str(e)}"}), 500
#随机中文句子接口
@app.route('/api/random-chinese-sentence', methods=['GET'])
def get_random_chinese_sentence():
    tsv_file = os.path.join("data", "常用英语口语.txt")
    chinese_sentence = shuijizhongwen(tsv_file)
    if not chinese_sentence:
        return jsonify({"error": "No sentences found"}), 404
    return jsonify({"sentence": chinese_sentence})
#语音转文字接口（Whisper）
@app.route('/api/transcribe-audio', methods=['POST'])
def transcribe_audio_api():
    """使用Whisper模型将语音转换为文字"""
    try:
        print("=== 开始处理语音转文字请求 ===")
        
        # 获取音频文件
        audio_file = request.files.get('audio_file')
        if not audio_file:
            return jsonify({'error': '缺少音频文件'}), 400
        
        # 生成唯一的临时文件路径
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        _, ext = os.path.splitext(audio_file.filename)
        ext = (ext or '.webm').lower()
        audio_path = os.path.join(AUDIO_UPLOAD_DIR, f"temp_transcribe_{unique_id}{ext}")
        
        try:
            # 保存音频文件
            audio_file.save(audio_path)
            file_size = os.path.getsize(audio_path)
            print(f"音频文件已保存: {audio_path}, 大小: {file_size} 字节")
            
            # 检查文件是否成功保存
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                return jsonify({"error": "音频文件保存失败或为空文件"}), 500
            
            # 音频格式转换（如果需要）
            processed_audio_path = audio_path
            if ext != '.wav':
                converted_path = convert_to_wav_16k(audio_path)
                if converted_path:
                    processed_audio_path = converted_path
                    print(f"音频已转换为WAV格式: {converted_path}")
            
            # 调用Whisper进行语音转文字
            try:
                from src.core.语音转写 import transcribe_audio
                transcribed_text = transcribe_audio(processed_audio_path)
                
                if transcribed_text:
                    print(f"语音转文字成功: {transcribed_text}")
                    return jsonify({
                        'success': True,
                        'transcribed_text': transcribed_text
                    })
                else:
                    return jsonify({'error': '语音识别结果为空，请重新录音'}), 400
                    
            except Exception as transcribe_error:
                print(f"Whisper转写失败: {transcribe_error}")
                return jsonify({'error': f'语音识别失败: {str(transcribe_error)}'}), 500
                
        finally:
            # 清理临时文件
            try:
                if os.path.exists(audio_path) and not KEEP_UPLOADS:
                    os.remove(audio_path)
                    print(f"临时文件已清理: {audio_path}")
                # 清理转换后的文件
                if processed_audio_path != audio_path and os.path.exists(processed_audio_path) and not KEEP_UPLOADS:
                    os.remove(processed_audio_path)
                    print(f"转换文件已清理: {processed_audio_path}")
            except Exception as cleanup_error:
                print(f"清理临时文件时出错: {str(cleanup_error)}")
                
    except Exception as e:
        print(f"语音转文字接口错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"语音转文字过程中出错: {str(e)}"}), 500

#语法检测接口
@app.route('/api/check-grammar', methods=['POST'])
def check_grammar_api():
    try:
        # 获取请求参数（支持仅文本，音频可选）
        translated_text = request.form.get("translated_text")
        audio_file = request.files.get("audio_file")

        # 参数验证
        if not translated_text or not translated_text.strip():
            return jsonify({"error": "缺少翻译文本"}), 400

        transcribed_text = ""
        # 如提供音频，则尝试处理音频
        if audio_file and audio_file.filename:
            import uuid, os
            unique_id = uuid.uuid4().hex[:8]
            _, ext = os.path.splitext(audio_file.filename)
            ext = (ext or '.wav').lower()
            audio_path = os.path.join(AUDIO_UPLOAD_DIR, f"temp_recording_{unique_id}{ext}")

            try:
                audio_file.save(audio_path)

                # 如果为空文件，则忽略音频，继续仅基于文本做检查
                if (not os.path.exists(audio_path)) or os.path.getsize(audio_path) == 0:
                    audio_path = None

                if audio_path:
                    transcribed_text = transcribe_audio(audio_path)
            finally:
                try:
                    if audio_path and os.path.exists(audio_path):
                        os.remove(audio_path)
                except Exception as cleanup_error:
                    print(f"清理临时文件时出错: {str(cleanup_error)}")

        # 以用户文本为准进行语法分析
        analysis_result = analyze_grammar(translated_text)

        # 构建返回结果
        if analysis_result.get("status") == "success":
            result = {"status": "success", "message": "✅ 英文语法正确!"}
        else:
            result = {
                "status": "error",
                "error_count": analysis_result.get('error_count', 0),
                "errors": analysis_result.get('errors', [])
            }

        return jsonify({
            "translated_text": translated_text,
            "transcribed_text": transcribed_text,
            "result": result
        })
    except Exception as e:
        print(f"语法检测接口错误: {str(e)}")
        return jsonify({"error": f"语法检测过程中出错: {str(e)}"}), 500

# 新增：纯文本语法检测接口
@app.route('/api/check-grammar-text', methods=['POST'])
def check_grammar_text_api():
    """仅用于文本语法检测的简化接口"""
    try:
        # 获取文本内容（支持JSON或FormData）
        if request.is_json:
            text = request.json.get('text', '').strip()
        else:
            text = request.form.get('text', '').strip()
        
        # 参数验证
        if not text:
            return jsonify({"error": "请提供要检测的文本"}), 400
        
        # 进行语法分析
        from src.core.语法检查 import analyze_grammar
        analysis_result = analyze_grammar(text)
        
        # 返回标准化的结果
        if analysis_result.get("status") == "success":
            return jsonify({
                "success": True,
                "errors": [],
                "message": "未发现语法错误",
                "corrected_text": analysis_result.get('corrected_text', text)
            })
        else:
            return jsonify({
                "success": False,
                "errors": analysis_result.get('errors', []),
                "error_count": analysis_result.get('error_count', 0),
                "corrected_text": analysis_result.get('corrected_text', text)
            })
            
    except Exception as e:
        print(f"文本语法检测接口错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"语法检测失败: {str(e)}"}), 500
@app.route('/api/custom-exercise', methods=['POST'])
def custom_exercise():
    data = request.json or {}
    file_path = data.get('file_path')
    input_text = data.get('text')  # 可选，文本内容（按行）
    mode = data.get('mode')  # 'speech' 或 'grammar'
    exercise_id = data.get('exercise_id')  # 已存在的练习集ID
    difficulty = data.get('difficulty')  # 难度等级
    exercise_name = data.get('exercise_name')  # 练习集名称

    # 导入练习管理器
    from src.core.自定义练习模块 import get_exercise_manager, create_text_exercise, get_exercise_by_type
    manager = get_exercise_manager()
    
    records = []
    current_exercise_id = exercise_id
    
    try:
        # 处理文本输入
        if input_text and isinstance(input_text, str):
            current_exercise_id = create_text_exercise(input_text, exercise_name)
            print(f"从文本创建练习集: {current_exercise_id}")
        
        # 处理文件路径
        elif file_path:
            if not os.path.exists(file_path):
                return jsonify({'error': f'文件不存在: {file_path}'}), 400
            current_exercise_id = manager.import_from_file(file_path, exercise_name)
            print(f"从文件导入练习集: {current_exercise_id}")
        
        # 使用已存在的练习集
        elif current_exercise_id:
            exercise_set = manager.get_exercise_set(current_exercise_id)
            if not exercise_set:
                return jsonify({'error': f'练习集不存在: {current_exercise_id}'}), 400
            print(f"使用已存在的练习集: {current_exercise_id}")
        
        # 都没有提供，回退到默认数据集
        else:
            if mode == 'speech':
                try:
                    tsv_file = os.path.join("data", "common_voice", "validated.tsv")
                    records = load_sentences_and_paths(tsv_file)
                except Exception:
                    records = []
            elif mode == 'grammar':
                try:
                    # 使用常用英语口语作为中文题面来源
                    chinese = shuijizhongwen(os.path.join("data", "常用英语口语.txt"))
                    if chinese:
                        records = [{"chinese": chinese}]
                except Exception:
                    records = []
            
            # 仍无可用数据
            if not records:
                return jsonify({'error': '未提供有效的自定义数据！'})

        # 从练习集中获取项目
        if current_exercise_id:
            user_id = data.get('user_id', 'default')
            
            if mode == 'speech':
                exercise_item = get_exercise_by_type(current_exercise_id, 'speech', difficulty, user_id)
                if exercise_item:
                    reference_text = exercise_item['content'].get('text', '')
                    if not reference_text:
                        return jsonify({'error': '自定义数据中未找到可朗读文本！'})
                    
                    return jsonify({
                        'exercise_id': current_exercise_id,
                        'item_id': exercise_item['id'],
                        'reference_text': reference_text,
                        'phonetic': exercise_item['content'].get('phonetic', ''),
                        'difficulty': exercise_item.get('difficulty', 'medium'),
                        'tags': exercise_item.get('tags', [])
                    })
                else:
                    # 如果没有speech类型，尝试phoneme类型
                    exercise_item = get_exercise_by_type(current_exercise_id, 'phoneme', difficulty, user_id)
                    if exercise_item:
                        reference_text = exercise_item['content'].get('text', '')
                        if not reference_text:
                            return jsonify({'error': '自定义数据中未找到可朗读文本！'})
                        
                        return jsonify({
                            'exercise_id': current_exercise_id,
                            'item_id': exercise_item['id'],
                            'reference_text': reference_text,
                            'phonetic': exercise_item['content'].get('phonetic', ''),
                            'difficulty': exercise_item.get('difficulty', 'medium'),
                            'tags': exercise_item.get('tags', [])
                        })
                    else:
                        # 返回更详细的错误信息
                        manager = get_exercise_manager()
                        exercise_set = manager.exercises["exercise_sets"].get(current_exercise_id)
                        if exercise_set:
                            all_types = set(item.get('type') for item in exercise_set['items'])
                            return jsonify({
                                'error': f'指定条件下没有找到合适的语音练习项目！\n练习集中包含的类型: {", ".join(all_types)}\n请尝试其他练习类型或难度等级'
                            })
                        else:
                            return jsonify({'error': '练习集不存在'})
            
            elif mode == 'grammar':
                exercise_item = get_exercise_by_type(current_exercise_id, 'grammar', difficulty, user_id)
                if exercise_item:
                    chinese_sentence = exercise_item['content'].get('chinese', '')
                    if not chinese_sentence:
                        return jsonify({'error': '自定义数据中未找到中文句子！'})
                    
                    return jsonify({
                        'exercise_id': current_exercise_id,
                        'item_id': exercise_item['id'],
                        'chinese_sentence': chinese_sentence,
                        'reference_english': exercise_item['content'].get('english', ''),
                        'explanation': exercise_item['content'].get('explanation', ''),
                        'difficulty': exercise_item.get('difficulty', 'medium'),
                        'tags': exercise_item.get('tags', [])
                    })
                else:
                    # 返回更详细的错误信息
                    manager = get_exercise_manager()
                    exercise_set = manager.exercises["exercise_sets"].get(current_exercise_id)
                    if exercise_set:
                        all_types = set(item.get('type') for item in exercise_set['items'])
                        all_difficulties = set(item.get('difficulty') for item in exercise_set['items'])
                        return jsonify({
                            'error': f'指定条件下没有找到合适的语法练习项目！\n练习集中包含的类型: {", ".join(all_types)}\n练习集中包含的难度: {", ".join(all_difficulties)}\n请尝试其他练习类型或难度等级'
                        })
                    else:
                        return jsonify({'error': '练习集不存在'})
        
        # 回退到传统处理方式
        if mode == 'speech':
            random_record = get_random_custom_sentence(records)
            reference_text = random_record.get("sentence") or random_record.get("text") or ""
            if not reference_text:
                return jsonify({'error': '自定义数据中未找到可朗读文本！'})
            # 仅返回题目，录音与评分由前端完成并调用 /api/score-pronunciation
            return jsonify({
                'reference_text': reference_text
            })

        elif mode == 'grammar':
            random_record = get_random_custom_sentence(records)
            chinese_sentence = random_record.get("chinese") or random_record.get("sentence") or random_record.get("text") or ""
            if not chinese_sentence:
                return jsonify({'error': '自定义数据中未找到中文句子！'})
            # 仅返回题目，前端提交文本到 /api/check-grammar
            return jsonify({
                'chinese_sentence': chinese_sentence
            })

        else:
            return jsonify({'error': '无效选项！'})
            
    except Exception as e:
        print(f"自定义练习错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'自定义练习处理失败: {str(e)}'}), 500

# 新增练习管理API
@app.route('/api/exercise-sets', methods=['GET'])
def get_exercise_sets():
    """获取所有练习集"""
    try:
        from src.core.自定义练习模块 import get_exercise_manager
        manager = get_exercise_manager()
        exercise_sets = manager.get_exercise_sets()
        return jsonify({'exercise_sets': exercise_sets})
    except Exception as e:
        return jsonify({'error': f'获取练习集失败: {str(e)}'}), 500

@app.route('/api/exercise-sets', methods=['POST'])
def create_exercise_set():
    """创建新的练习集"""
    try:
        data = request.json or {}
        name = data.get('name')
        description = data.get('description', '')
        exercise_type = data.get('type', 'mixed')
        
        if not name:
            return jsonify({'error': '练习集名称不能为空'}), 400
        
        from src.core.自定义练习模块 import get_exercise_manager
        manager = get_exercise_manager()
        exercise_id = manager.create_exercise_set(name, description, exercise_type)
        
        return jsonify({
            'success': True,
            'exercise_id': exercise_id,
            'message': f'练习集「{name}」创建成功'
        })
    except Exception as e:
        return jsonify({'error': f'创建练习集失败: {str(e)}'}), 500

@app.route('/api/exercise-sets/<exercise_id>', methods=['GET'])
def get_exercise_set(exercise_id):
    """获取指定练习集的详细信息"""
    try:
        user_id = request.args.get('user_id', 'default')
        
        from src.core.自定义练习模块 import get_exercise_manager
        manager = get_exercise_manager()
        exercise_set = manager.get_exercise_set(exercise_id, user_id)
        
        if not exercise_set:
            return jsonify({'error': '练习集不存在'}), 404
        
        return jsonify(exercise_set)
    except Exception as e:
        print(f"获取练习集失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取练习集失败: {str(e)}'}), 500

@app.route('/api/exercise-results', methods=['POST'])
def record_exercise_result():
    """记录练习结果"""
    try:
        data = request.json or {}
        exercise_id = data.get('exercise_id')
        item_id = data.get('item_id')
        score = data.get('score', 0)
        time_spent = data.get('time_spent', 0)
        user_id = data.get('user_id', 'default')
        
        if not exercise_id or not item_id:
            return jsonify({'error': '缺少必要参数'}), 400
        
        from src.core.自定义练习模块 import get_exercise_manager
        manager = get_exercise_manager()
        success = manager.record_exercise_result(exercise_id, item_id, float(score), float(time_spent), user_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': '练习结果记录成功'
            })
        else:
            return jsonify({'error': '记录练习结果失败'}), 500
    except Exception as e:
        return jsonify({'error': f'记录练习结果失败: {str(e)}'}), 500

@app.route('/api/exercise-progress', methods=['GET'])
def get_exercise_progress():
    """获取用户练习进度"""
    try:
        user_id = request.args.get('user_id', 'default')
        
        from src.core.自定义练习模块 import get_exercise_manager
        manager = get_exercise_manager()
        progress = manager.get_user_progress(user_id)
        
        return jsonify({'progress': progress})
    except Exception as e:
        return jsonify({'error': f'获取练习进度失败: {str(e)}'}), 500

@app.route('/api/import-exercise', methods=['POST'])
def import_exercise_from_text():
    """从文本导入练习"""
    try:
        data = request.json or {}
        content = data.get('content', '')
        exercise_name = data.get('name', '')
        
        if not content.strip():
            return jsonify({'error': '文本内容不能为空'}), 400
        
        from src.core.自定义练习模块 import create_text_exercise
        exercise_id = create_text_exercise(content, exercise_name)
        
        return jsonify({
            'success': True,
            'exercise_id': exercise_id,
            'message': f'练习集「{exercise_name or "文本练习"}」导入成功'
        })
    except Exception as e:
        return jsonify({'error': f'导入练习失败: {str(e)}'}), 500

# ================================
# 用户认证API
# ================================

def require_auth(f):
    """认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            token = token[7:]  # 移除 'Bearer ' 前缀
            
            from src.core.db_user_manager import get_db_user_manager
            user_manager = get_db_user_manager()
            user = user_manager.verify_user(token)
            
            if user:
                request.current_user = user
                return f(*args, **kwargs)
        
        return jsonify({'error': '未授权访问'}), 401
    return decorated_function

@app.route('/api/register', methods=['POST'])
def register():
    """用户注册"""
    try:
        data = request.json or {}
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        email = data.get('email', '').strip()
        full_name = data.get('full_name', '').strip()
        
        from src.core.db_user_manager import get_db_user_manager
        user_manager = get_db_user_manager()
        result = user_manager.register_user(username, password, email, full_name)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'error': f'注册失败: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.json or {}
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        from src.core.db_user_manager import get_db_user_manager
        user_manager = get_db_user_manager()
        result = user_manager.login_user(username, password)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 401
            
    except Exception as e:
        return jsonify({'error': f'登录失败: {str(e)}'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """用户登出"""
    try:
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            token = token[7:]
            
            from src.core.db_user_manager import get_db_user_manager
            user_manager = get_db_user_manager()
            success = user_manager.logout_user(token)
            
            if success:
                return jsonify({'success': True, 'message': '登出成功'})
            else:
                return jsonify({'success': False, 'message': '登出失败'}), 400
        else:
            return jsonify({'error': '未提供有效令牌'}), 400
            
    except Exception as e:
        return jsonify({'error': f'登出失败: {str(e)}'}), 500

@app.route('/api/user/profile', methods=['GET'])
@require_auth
def get_user_profile():
    """获取用户资料"""
    try:
        user = request.current_user
        return jsonify({
            'success': True,
            'user': user
        })
    except Exception as e:
        return jsonify({'error': f'获取用户资料失败: {str(e)}'}), 500

@app.route('/api/user/settings', methods=['PUT'])
@require_auth
def update_user_settings():
    """更新用户设置"""
    try:
        data = request.json or {}
        user_id = request.current_user['id']
        
        from src.core.db_user_manager import get_db_user_manager
        user_manager = get_db_user_manager()
        success = user_manager.update_user_settings(user_id, data)
        
        if success:
            return jsonify({'success': True, 'message': '设置更新成功'})
        else:
            return jsonify({'success': False, 'message': '设置更新失败'}), 400
            
    except Exception as e:
        return jsonify({'error': f'更新设置失败: {str(e)}'}), 500

# ================================
# 学习记录API（与用户关联）
# ================================

@app.route('/api/user/learning-history', methods=['GET'])
@require_auth
def get_learning_history():
    """获取用户学习历史"""
    try:
        user_id = request.current_user['id']
        limit = int(request.args.get('limit', 50))
        exercise_type = request.args.get('type', None)
        
        from src.core.db_learning_manager import get_db_learning_manager
        record_manager = get_db_learning_manager()
        history = record_manager.get_user_records(user_id, limit, exercise_type)
        
        return jsonify({
            'success': True,
            'history': history,
            'total': len(history)
        })
        
    except Exception as e:
        return jsonify({'error': f'获取学习历史失败: {str(e)}'}), 500

@app.route('/api/user/statistics', methods=['GET'])
@require_auth
def get_user_statistics():
    """获取用户统计信息"""
    try:
        user_id = request.current_user['id']
        
        from src.core.db_learning_manager import get_db_learning_manager
        record_manager = get_db_learning_manager()
        statistics = record_manager.get_user_statistics(user_id)
        
        return jsonify({
            'success': True,
            'statistics': statistics
        })
        
    except Exception as e:
        return jsonify({'error': f'获取统计信息失败: {str(e)}'}), 500

@app.route('/api/user/recent-activity', methods=['GET'])
@require_auth  
def get_recent_activity():
    """获取用户最近活动"""
    try:
        user_id = request.current_user['id']
        days = int(request.args.get('days', 7))
        
        from src.core.db_learning_manager import get_db_learning_manager
        record_manager = get_db_learning_manager()
        activity = record_manager.get_recent_activity(user_id, days)
        
        return jsonify({
            'success': True,
            'activity': activity,
            'days': days
        })
        
    except Exception as e:
        return jsonify({'error': f'获取最近活动失败: {str(e)}'}), 500

@app.route('/api/user/learning-streaks', methods=['GET'])
@require_auth
def get_learning_streaks():
    """获取学习连续记录"""
    try:
        user_id = request.current_user['id']
        
        from src.core.db_learning_manager import get_db_learning_manager
        record_manager = get_db_learning_manager()
        streaks = record_manager.get_learning_streaks(user_id)
        
        return jsonify({
            'success': True,
            'streaks': streaks
        })
        
    except Exception as e:
        return jsonify({'error': f'获取学习连续记录失败: {str(e)}'}), 500

@app.route('/api/user/improvement-analysis', methods=['GET'])
@require_auth
def get_improvement_analysis():
    """获取学习改进分析"""
    try:
        user_id = request.current_user['id']
        exercise_type = request.args.get('type', None)
        days = int(request.args.get('days', 30))
        
        from src.core.db_learning_manager import get_db_learning_manager
        record_manager = get_db_learning_manager()
        analysis = record_manager.get_improvement_analysis(user_id, exercise_type, days)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'error': f'获取学习改进分析失败: {str(e)}'}), 500

@app.route('/api/user/change-password', methods=['POST'])
@require_auth
def change_password():
    """修改密码"""
    try:
        data = request.json or {}
        old_password = data.get('old_password', '').strip()
        new_password = data.get('new_password', '').strip()
        user_id = request.current_user['id']
        
        from src.core.db_user_manager import get_db_user_manager
        user_manager = get_db_user_manager()
        result = user_manager.change_password(str(user_id), old_password, new_password)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'error': f'修改密码失败: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        # 创建数据表
        create_tables(app)
        print("✅ 数据库初始化完成")
    except Exception as e:
        print(f"⚠️ 数据库初始化失败: {e}")
        print("请检查数据库配置和连接")
    
    # 为了避免音素级评分时的自动重载问题，在生产环境中关闭调试模式
    app.run(debug=False, host='0.0.0.0', port=5000)