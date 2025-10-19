import numpy as np
import torch
import librosa
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
import traceback

@dataclass
class PhonemeScore:
    """音素评分结果"""
    phoneme: str          # 音素符号
    start_time: float     # 开始时间(秒)
    end_time: float       # 结束时间(秒)
    score: float          # 评分(0-100)
    confidence: float     # 置信度(0-1)
    quality: str          # 质量等级(excellent/good/fair/poor)
    issues: List[str]     # 发音问题列表

@dataclass
class DetailedPronunciationResult:
    """详细发音评分结果"""
    overall_score: float                    # 总分(0-100)
    phoneme_scores: List[PhonemeScore]      # 音素级评分
    word_scores: List[Dict]                 # 单词级评分
    pronunciation_issues: List[str]         # 发音问题总结
    improvement_suggestions: List[str]      # 改进建议
    duration_analysis: Dict                 # 时长分析
    pitch_analysis: Dict                    # 语调分析

class PhonemeScorer:
    """音素级发音评分器"""
    
    def __init__(self):
        self.phoneme_map = self._load_phoneme_map()
        self.duration_thresholds = self._load_duration_thresholds()
        
    def _load_phoneme_map(self) -> Dict[str, str]:
        """加载音素映射表（文本到IPA音素）"""
        # 简化的音素映射，实际应用中需要更完整的映射
        return {
            # 元音
            'a': 'æ', 'e': 'e', 'i': 'ɪ', 'o': 'ɒ', 'u': 'ʊ',
            'ah': 'ʌ', 'ay': 'aɪ', 'aw': 'aʊ', 'oy': 'ɔɪ',
            'ee': 'iː', 'oo': 'uː', 'er': 'ɜː', 'ar': 'ɑː',
            
            # 辅音
            'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
            'f': 'f', 'v': 'v', 'th': 'θ', 'dh': 'ð', 's': 's', 'z': 'z',
            'sh': 'ʃ', 'zh': 'ʒ', 'ch': 'tʃ', 'j': 'dʒ',
            'm': 'm', 'n': 'n', 'ng': 'ŋ', 'l': 'l', 'r': 'r',
            'w': 'w', 'y': 'j', 'h': 'h'
        }
    
    def _load_duration_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """加载音素时长阈值"""
        return {
            # 格式: 音素: (最小时长, 最大时长) 单位：秒
            'æ': (0.08, 0.15), 'ɪ': (0.06, 0.12), 'ʊ': (0.06, 0.12),
            'iː': (0.10, 0.20), 'uː': (0.10, 0.20), 'ɜː': (0.12, 0.25),
            'p': (0.02, 0.08), 'b': (0.03, 0.10), 't': (0.02, 0.08),
            'd': (0.03, 0.10), 'k': (0.02, 0.08), 'g': (0.03, 0.10),
            'f': (0.08, 0.15), 'v': (0.06, 0.12), 's': (0.08, 0.18),
            'z': (0.06, 0.15), 'ʃ': (0.08, 0.16), 'ʒ': (0.06, 0.14),
        }
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """将文本转换为音素序列（简化版本）"""
        # 这是一个简化的实现，实际应用中需要使用专业的phonemizer
        text = text.lower().strip()
        words = text.split()
        phonemes = []
        
        # 基础的音素转换规则
        phoneme_rules = {
            'the': ['ð', 'ə'],
            'and': ['æ', 'n', 'd'],
            'have': ['h', 'æ', 'v'],
            'that': ['ð', 'æ', 't'],
            'for': ['f', 'ɔː'],
            'are': ['ɑː'],
            'with': ['w', 'ɪ', 'θ'],
            'his': ['h', 'ɪ', 'z'],
            'they': ['ð', 'eɪ'],
            'this': ['ð', 'ɪ', 's'],
            'from': ['f', 'r', 'ɒ', 'm'],
            'she': ['ʃ', 'iː'],
            'her': ['h', 'ɜː'],
            'been': ['b', 'iː', 'n'],
            'than': ['ð', 'æ', 'n'],
            'its': ['ɪ', 't', 's'],
            'now': ['n', 'aʊ'],
            'more': ['m', 'ɔː'],
            'very': ['v', 'e', 'r', 'ɪ'],
            'what': ['w', 'ɒ', 't'],
            'know': ['n', 'əʊ'],
            'just': ['dʒ', 'ʌ', 's', 't'],
            'first': ['f', 'ɜː', 's', 't'],
            'time': ['t', 'aɪ', 'm'],
            'people': ['p', 'iː', 'p', 'ə', 'l'],
        }
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word)
            if word_clean in phoneme_rules:
                phonemes.extend(phoneme_rules[word_clean])
            else:
                # 简单的字母到音素映射
                for char in word_clean:
                    if char in self.phoneme_map:
                        phonemes.append(self.phoneme_map[char])
                    else:
                        phonemes.append(char)  # 保留未知字符
        
        return phonemes
    
    def extract_acoustic_features(self, audio_data: np.ndarray, sr: int = 16000) -> Dict:
        """提取声学特征"""
        try:
            features = {}
            
            # 基频(F0)提取
            f0 = librosa.yin(audio_data, fmin=80, fmax=400, sr=sr)
            features['f0'] = f0
            features['f0_mean'] = np.nanmean(f0[f0 > 0]) if np.any(f0 > 0) else 0
            features['f0_std'] = np.nanstd(f0[f0 > 0]) if np.any(f0 > 0) else 0
            
            # MFCC特征
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            features['mfcc'] = mfcc
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # 频谱质心和带宽
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            
            # 能量特征
            rms = librosa.feature.rms(y=audio_data)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
            # 零交叉率
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = np.mean(zcr)
            
            return features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return {}
    
    def force_align_ctc(self, audio_data: np.ndarray, phoneme_sequence: List[str], 
                       wav2vec2_model, processor, sr: int = 16000) -> List[Tuple[str, float, float]]:
        """使用Wav2Vec2 CTC进行强制对齐"""
        try:
            # 预处理音频
            inputs = processor(audio_data, sampling_rate=sr, return_tensors="pt", padding=True)
            
            # 获取CTC输出
            with torch.no_grad():
                logits = wav2vec2_model(**inputs).logits
            
            # 获取预测序列
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # 简化的对齐：均匀分割时间
            # 实际应用中需要更复杂的CTC对齐算法
            total_duration = len(audio_data) / sr
            phoneme_count = len(phoneme_sequence)
            
            if phoneme_count == 0:
                return []
            
            avg_duration = total_duration / phoneme_count
            alignments = []
            
            for i, phoneme in enumerate(phoneme_sequence):
                start_time = i * avg_duration
                end_time = (i + 1) * avg_duration
                alignments.append((phoneme, start_time, end_time))
            
            return alignments
            
        except Exception as e:
            print(f"强制对齐失败: {e}")
            # 返回均匀分割的对齐结果
            total_duration = len(audio_data) / sr
            phoneme_count = len(phoneme_sequence)
            
            if phoneme_count == 0:
                return []
            
            avg_duration = total_duration / phoneme_count
            alignments = []
            
            for i, phoneme in enumerate(phoneme_sequence):
                start_time = i * avg_duration
                end_time = (i + 1) * avg_duration
                alignments.append((phoneme, start_time, end_time))
            
            return alignments
    
    def score_phoneme_quality(self, phoneme: str, features: Dict, duration: float) -> Tuple[float, List[str]]:
        """评估单个音素的发音质量（更加严格的评分标准）"""
        score = 80.0  # 降低基础分数，使评分更加严格
        issues = []
        
        # 更严格的时长评估
        if phoneme in self.duration_thresholds:
            min_dur, max_dur = self.duration_thresholds[phoneme]
            if duration < min_dur * 0.7:  # 更严格的下限
                score -= 30
                issues.append(f"音素'{phoneme}'发音过短，需要更充分的发声")
            elif duration < min_dur:
                score -= 20
                issues.append(f"音素'{phoneme}'发音略短")
            elif duration > max_dur * 1.5:  # 更严格的上限
                score -= 25
                issues.append(f"音素'{phoneme}'发音过长，注意控制节奏")
            elif duration > max_dur:
                score -= 15
                issues.append(f"音素'{phoneme}'发音略长")
        
        # 基于MFCC的质量评估（更严格）
        if 'mfcc_mean' in features and len(features['mfcc_mean']) > 0:
            mfcc_stability = np.std(features['mfcc_mean'])
            if mfcc_stability > 30:  # 降低阈值，更严格
                score -= 20
                issues.append(f"音素'{phoneme}'发音不稳定，可能存在紧张或不确定")
            elif mfcc_stability > 20:
                score -= 10
                issues.append(f"音素'{phoneme}'发音稍显不稳定")
        
        # 更严格的能量评估
        if 'energy_mean' in features:
            if features['energy_mean'] < 0.005:  # 提高阈值
                score -= 25
                issues.append(f"音素'{phoneme}'发音能量不足，需要更加清晰有力的发声")
            elif features['energy_mean'] < 0.01:
                score -= 15
                issues.append(f"音素'{phoneme}'发音能量较低")
        
        # 更严格的频谱质心评估（用于判断清晰度）
        if 'spectral_centroid_mean' in features:
            # 不同音素有不同的频谱特征期望值
            if phoneme in ['s', 'ʃ', 'f', 'θ']:  # 高频摩擦音
                if features['spectral_centroid_mean'] < 2500:  # 提高标准
                    score -= 20
                    issues.append(f"高频摩擦音'{phoneme}'高频成分不足，需要更明显的摩擦声")
            elif phoneme in ['æ', 'ɑː', 'ɒ']:  # 低频元音
                if features['spectral_centroid_mean'] > 1500:  # 提高标准
                    score -= 15
                    issues.append(f"低频元音'{phoneme}'音色偏高，需要更低的舌位")
            elif phoneme in ['iː', 'ɪ']:  # 高元音
                if features['spectral_centroid_mean'] < 1000 or features['spectral_centroid_mean'] > 2200:
                    score -= 15
                    issues.append(f"高元音'{phoneme}'舌位不准确，需要调整口型")
        
        # 新增：更细致的音素类别检查
        phoneme_type = self.classify_phoneme_detailed(phoneme)
        type_issues = self.check_phoneme_type_quality(phoneme, phoneme_type, features, duration)
        issues.extend(type_issues)
        score -= len(type_issues) * 8  # 每个类型问题扣8分
        
        return max(0, min(100, score)), issues
    
    def classify_phoneme_detailed(self, phoneme: str) -> str:
        """更详细的音素分类"""
        vowels = ['æ', 'ɪ', 'ʊ', 'iː', 'uː', 'ɜː', 'ʌ', 'aɪ', 'aʊ', 'ɔɪ', 'e', 'ɒ', 'ɑː']
        fricatives = ['f', 'v', 's', 'z', 'ʃ', 'ʒ', 'θ', 'ð', 'h']
        stops = ['p', 'b', 't', 'd', 'k', 'g']
        affricates = ['tʃ', 'dʒ']
        nasals = ['m', 'n', 'ŋ']
        liquids = ['l', 'r']
        glides = ['w', 'j']
        
        if phoneme in vowels:
            # 进一步细分元音
            if phoneme in ['æ', 'ɑː', 'ɒ']:
                return 'low_vowel'  # 低元音
            elif phoneme in ['iː', 'ɪ', 'uː', 'ʊ']:
                return 'high_vowel'  # 高元音
            elif phoneme in ['e', 'ɜː', 'ʌ']:
                return 'mid_vowel'  # 中元音
            else:
                return 'diphthong'  # 双元音
        elif phoneme in fricatives:
            if phoneme in ['s', 'z', 'ʃ', 'ʒ']:
                return 'sibilant_fricative'  # 喙音摩擦音
            else:
                return 'non_sibilant_fricative'  # 非喙音摩擦音
        elif phoneme in stops:
            if phoneme in ['p', 't', 'k']:
                return 'voiceless_stop'  # 清音爆破音
            else:
                return 'voiced_stop'  # 浊音爆破音
        elif phoneme in affricates:
            return 'affricate'  # 塞擦音
        elif phoneme in nasals:
            return 'nasal'
        elif phoneme in liquids:
            return 'liquid'
        elif phoneme in glides:
            return 'glide'
        else:
            return 'unknown'
    
    def check_phoneme_type_quality(self, phoneme: str, phoneme_type: str, features: Dict, duration: float) -> List[str]:
        """检查特定音素类型的质量问题"""
        issues = []
        
        # 低元音检查
        if phoneme_type == 'low_vowel':
            if 'f1' in features and features['f1'] > 0:
                if features['f1'] < 600:  # F1应该较高
                    issues.append(f"低元音'{phoneme}'开口度不够，需要更大的张口")
        
        # 高元音检查
        elif phoneme_type == 'high_vowel':
            if 'f1' in features and features['f1'] > 0:
                if features['f1'] > 450:  # F1应该较低
                    issues.append(f"高元音'{phoneme}'舌位过低，需要提高舌位")
        
        # 喙音摩擦音检查
        elif phoneme_type == 'sibilant_fricative':
            if 'zcr_mean' in features:
                if features['zcr_mean'] < 0.15:  # 零交叉率应该较高
                    issues.append(f"喙音'{phoneme}'摩擦声不够明显，需要更明显的气流摩擦")
        
        # 清音爆破音检查
        elif phoneme_type == 'voiceless_stop':
            if 'energy_max' in features and 'energy_mean' in features:
                if features['energy_mean'] > 0:
                    energy_ratio = features['energy_max'] / features['energy_mean']
                    if energy_ratio < 2.5:  # 爆破特征不明显
                        issues.append(f"清音爆破音'{phoneme}'爆破特征不明显，需要更明显的爆破壴")
        
        # 浊音爆破音检查
        elif phoneme_type == 'voiced_stop':
            if 'voicing_rate' in features:
                if features['voicing_rate'] < 0.6:  # 浊音程度不足
                    issues.append(f"浊音爆破音'{phoneme}'浊音程度不足，需要更多声带振动")
        
        # 鼻音检查
        elif phoneme_type == 'nasal':
            if 'f1' in features and features['f1'] > 0:
                if features['f1'] > 500:  # 鼻音F1应该较低
                    issues.append(f"鼻音'{phoneme}'鼻腔共鸣不足，注意软腊下降")
        
        return issues
    
    def get_quality_level(self, score: float) -> str:
        """根据评分获取质量等级"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        else:
            return "poor"
    
    def analyze_pronunciation_detailed(self, audio_data: np.ndarray, reference_text: str,
                                     wav2vec2_model, processor, sr: int = 16000) -> DetailedPronunciationResult:
        """执行详细的发音分析"""
        try:
            print(f"开始音素级发音分析: '{reference_text}'")
            
            # 1. 文本转音素
            phoneme_sequence = self.text_to_phonemes(reference_text)
            print(f"音素序列: {phoneme_sequence}")
            
            # 2. 单词分割和音素映射
            words = reference_text.lower().split()
            word_phoneme_mapping = self.map_words_to_phonemes(words)
            print(f"单词音素映射: {word_phoneme_mapping}")
            
            # 3. 强制对齐
            alignments = self.force_align_ctc(audio_data, phoneme_sequence, wav2vec2_model, processor, sr)
            print(f"对齐结果数量: {len(alignments)}")
            
            # 4. 提取整体声学特征
            global_features = self.extract_acoustic_features(audio_data, sr)
            
            # 5. 音素级评分
            phoneme_scores = []
            word_scores = []
            all_issues = []
            
            for phoneme, start_time, end_time in alignments:
                # 提取该音素对应的音频段
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                phoneme_audio = audio_data[start_sample:end_sample]
                
                if len(phoneme_audio) > 0:
                    # 提取音素级特征
                    phoneme_features = self.extract_acoustic_features(phoneme_audio, sr)
                    
                    # 评分
                    duration = end_time - start_time
                    score, issues = self.score_phoneme_quality(phoneme, phoneme_features, duration)
                    
                    # 计算置信度（基于特征稳定性）
                    confidence = min(1.0, max(0.3, 1.0 - len(issues) * 0.1))
                    
                    phoneme_score = PhonemeScore(
                        phoneme=phoneme,
                        start_time=start_time,
                        end_time=end_time,
                        score=score,
                        confidence=confidence,
                        quality=self.get_quality_level(score),
                        issues=issues
                    )
                    
                    phoneme_scores.append(phoneme_score)
                    all_issues.extend(issues)
            
            # 6. 单词级评分和分析
            word_scores = self.analyze_word_pronunciation(words, word_phoneme_mapping, phoneme_scores)
            
            # 7. 计算总分（更加严格的评分标准）
            if phoneme_scores:
                # 加权平均，考虑置信度和质量等级
                weighted_scores = []
                quality_weights = {'excellent': 1.0, 'good': 0.9, 'fair': 0.7, 'poor': 0.4}
                
                for ps in phoneme_scores:
                    # 结合评分、置信度和质量等级
                    quality_weight = quality_weights.get(ps.quality, 0.5)
                    adjusted_score = ps.score * ps.confidence * quality_weight
                    weighted_scores.append(adjusted_score)
                
                # 计算基础平均分
                base_score = np.mean(weighted_scores) if weighted_scores else 0
                
                # 应用额外的严格度惩罚
                penalty = 0
                
                # 问题音素比例惩罚
                poor_ratio = len([ps for ps in phoneme_scores if ps.quality == 'poor']) / len(phoneme_scores)
                fair_ratio = len([ps for ps in phoneme_scores if ps.quality == 'fair']) / len(phoneme_scores)
                
                if poor_ratio > 0.3:  # 超过30%的音素质量差
                    penalty += 15
                elif poor_ratio > 0.2:
                    penalty += 10
                
                if fair_ratio > 0.4:  # 超过40%的音素质量一般
                    penalty += 8
                
                # 问题数量惩罚
                total_issues = len(all_issues)
                if total_issues > len(phoneme_scores) * 0.5:  # 平均每个音素超过0.5个问题
                    penalty += 12
                elif total_issues > len(phoneme_scores) * 0.3:
                    penalty += 6
                
                # 严重问题额外惩罚
                severe_issues = [issue for issue in all_issues if '过短' in issue or '过长' in issue or '能量不足' in issue]
                if len(severe_issues) > 3:
                    penalty += 10
                
                # 计算最终评分
                overall_score = max(0, min(100, base_score - penalty))
                
                # 如果评分仍然过高，额外调整
                if overall_score > 85 and (poor_ratio > 0.1 or len(severe_issues) > 1):
                    overall_score = min(85, overall_score - 5)
                
                if overall_score > 75 and poor_ratio > 0.2:
                    overall_score = min(75, overall_score - 5)
            else:
                overall_score = 0
            
            # 8. 分析语调和时长
            duration_analysis = {
                'total_duration': len(audio_data) / sr,
                'speech_rate': len(phoneme_sequence) / (len(audio_data) / sr) if len(audio_data) > 0 else 0,
                'avg_phoneme_duration': np.mean([ps.end_time - ps.start_time for ps in phoneme_scores]) if phoneme_scores else 0
            }
            
            pitch_analysis = {}
            if 'f0_mean' in global_features:
                pitch_analysis = {
                    'average_f0': global_features['f0_mean'],
                    'f0_variation': global_features['f0_std'],
                    'pitch_range': 'normal'  # 需要更复杂的分析
                }
            
            # 9. 生成改进建议（包括单词级建议）
            improvement_suggestions = self._generate_detailed_suggestions(
                phoneme_scores, word_scores, all_issues, reference_text
            )
            
            result = DetailedPronunciationResult(
                overall_score=overall_score,
                phoneme_scores=phoneme_scores,
                word_scores=word_scores,
                pronunciation_issues=list(set(all_issues)),
                improvement_suggestions=improvement_suggestions,
                duration_analysis=duration_analysis,
                pitch_analysis=pitch_analysis
            )
            
            print(f"音素级分析完成，总分: {overall_score:.1f}")
            return result
            
        except Exception as e:
            print(f"音素级分析失败: {e}")
            traceback.print_exc()
            
            # 返回默认结果
            return DetailedPronunciationResult(
                overall_score=0,
                phoneme_scores=[],
                word_scores=[],
                pronunciation_issues=[f"分析过程中出错: {str(e)}"],
                improvement_suggestions=["请检查音频质量并重新录音"],
                duration_analysis={'total_duration': len(audio_data) / sr if len(audio_data) > 0 else 0},
                pitch_analysis={}
            )
    
    def _generate_detailed_suggestions(self, phoneme_scores: List[PhonemeScore], word_scores: List[Dict], 
                                     all_issues: List[str], reference_text: str) -> List[str]:
        """生成包含单词级建议的详细改进建议"""
        suggestions = []
        
        # 1. 单词级别的建议（优先显示）
        words_needing_improvement = [ws for ws in word_scores if ws.get('needs_improvement', False)]
        if words_needing_improvement:
            suggestions.append("🎯 需要重点改进的单词:")
            for word_info in words_needing_improvement[:5]:  # 最多显示5个
                word = word_info['word']
                score = word_info['score']
                quality = word_info['quality']
                quality_desc = {'excellent': '优秀', 'good': '良好', 'fair': '一般', 'poor': '较差', 'unknown': '未知'}[quality]
                suggestions.append(f"  • '{word}' (评分: {score}, 质量: {quality_desc})")
                
                # 添加该单词的具体建议
                word_suggestions = word_info.get('suggestions', [])
                for ws in word_suggestions[:2]:  # 每个单词最多2个建议
                    suggestions.append(f"    - {ws}")
        
        # 2. 音素级别的总体问题分析
        poor_phonemes = [ps.phoneme for ps in phoneme_scores if ps.quality == 'poor']
        fair_phonemes = [ps.phoneme for ps in phoneme_scores if ps.quality == 'fair']
        
        if poor_phonemes:
            unique_poor = list(set(poor_phonemes))
            suggestions.append(f"\n🔍 需要重点练习的音素: {', '.join(unique_poor[:6])}")
        
        if fair_phonemes and not poor_phonemes:
            unique_fair = list(set(fair_phonemes))
            suggestions.append(f"\n📈 可以进一步完善的音素: {', '.join(unique_fair[:4])}")
        
        # 3. 按问题类型分类建议
        timing_issues = [issue for issue in all_issues if ('过短' in issue or '过长' in issue)]
        clarity_issues = [issue for issue in all_issues if ('不稳定' in issue or '清晰度' in issue or '能量不足' in issue)]
        articulation_issues = [issue for issue in all_issues if ('摩擦' in issue or '爆破' in issue or '舌位' in issue or '开口' in issue)]
        
        if timing_issues:
            suggestions.append("\n⏰ 时长控制建议:")
            suggestions.append("  • 注意控制每个音素的发音时长，避免过快或过慢")
            suggestions.append("  • 可以尝试跟读标准音频，培养节奏感")
        
        if clarity_issues:
            suggestions.append("\n🎤 清晰度改进建议:")
            suggestions.append("  • 提高发音清晰度，确保每个音素都发音充分")
            suggestions.append("  • 练习时可以适当放慢语速，保证准确性")
        
        if articulation_issues:
            suggestions.append("\n👄 发音技巧建议:")
            suggestions.append("  • 注意口型和舌位的准确性")
            suggestions.append("  • 对照镜子练习，观察口型变化")
        
        # 4. 整体练习建议
        overall_score = np.mean([ps.score for ps in phoneme_scores]) if phoneme_scores else 0
        
        if overall_score < 60:
            suggestions.append("\n📚 基础练习建议:")
            suggestions.append("  • 建议从基础音素开始，逐个攻克")
            suggestions.append("  • 多听标准发音，培养语感")
            suggestions.append("  • 每天坚持练习15-20分钟")
        elif overall_score < 75:
            suggestions.append("\n🎯 提升练习建议:")
            suggestions.append("  • 重点关注上述问题音素和单词")
            suggestions.append("  • 可以尝试录音对比，找出差距")
        elif overall_score < 85:
            suggestions.append("\n✨ 精进练习建议:")
            suggestions.append("  • 继续保持练习，注意细节完善")
            suggestions.append("  • 可以尝试更具挑战性的句子")
        else:
            suggestions.append("\n🌟 继续保持:")
            suggestions.append("  • 发音水平很好，继续保持！")
            suggestions.append("  • 可以尝试更复杂的语音练习")
        
        # 5. 实用练习技巧
        if len(all_issues) > 3:
            suggestions.append("\n💡 练习技巧提示:")
            suggestions.append("  • 建议使用慢速播放功能仔细听标准发音")
            suggestions.append("  • 可以分段练习，先掌握单个单词再连接成句")
            suggestions.append("  • 录制自己的发音并与标准发音对比")
        
        return suggestions
    
    def map_words_to_phonemes(self, words: List[str]) -> Dict[str, List[str]]:
        """将单词映射到对应的音素序列"""
        word_phoneme_mapping = {}
        
        # 扩展的音素规则，包含更多常用单词
        phoneme_rules = {
            'the': ['ð', 'ə'],
            'and': ['æ', 'n', 'd'],
            'have': ['h', 'æ', 'v'],
            'that': ['ð', 'æ', 't'],
            'for': ['f', 'ɔː'],
            'are': ['ɑː'],
            'with': ['w', 'ɪ', 'θ'],
            'his': ['h', 'ɪ', 'z'],
            'they': ['ð', 'eɪ'],
            'this': ['ð', 'ɪ', 's'],
            'from': ['f', 'r', 'ɒ', 'm'],
            'she': ['ʃ', 'iː'],
            'her': ['h', 'ɜː'],
            'been': ['b', 'iː', 'n'],
            'than': ['ð', 'æ', 'n'],
            'its': ['ɪ', 't', 's'],
            'now': ['n', 'aʊ'],
            'more': ['m', 'ɔː'],
            'very': ['v', 'e', 'r', 'ɪ'],
            'what': ['w', 'ɒ', 't'],
            'know': ['n', 'əʊ'],
            'just': ['dʒ', 'ʌ', 's', 't'],
            'first': ['f', 'ɜː', 's', 't'],
            'time': ['t', 'aɪ', 'm'],
            'people': ['p', 'iː', 'p', 'ə', 'l'],
            'good': ['g', 'ʊ', 'd'],
            'work': ['w', 'ɜː', 'k'],
            'school': ['s', 'k', 'uː', 'l'],
            'world': ['w', 'ɜː', 'l', 'd'],
            'great': ['g', 'r', 'eɪ', 't'],
            'think': ['θ', 'ɪ', 'ŋ', 'k'],
            'way': ['w', 'eɪ'],
            'make': ['m', 'eɪ', 'k'],
            'today': ['t', 'ə', 'd', 'eɪ'],
            'help': ['h', 'e', 'l', 'p'],
            'home': ['h', 'əʊ', 'm'],
            'nice': ['n', 'aɪ', 's'],
            'happy': ['h', 'æ', 'p', 'ɪ'],
            'love': ['l', 'ʌ', 'v'],
            'like': ['l', 'aɪ', 'k'],
            'want': ['w', 'ɒ', 'n', 't'],
            'need': ['n', 'iː', 'd'],
            'thank': ['θ', 'æ', 'ŋ', 'k'],
            'you': ['j', 'uː'],
            'hello': ['h', 'ə', 'l', 'əʊ'],
            'water': ['w', 'ɔː', 't', 'ər'],
            'food': ['f', 'uː', 'd'],
            'money': ['m', 'ʌ', 'n', 'ɪ'],
            'house': ['h', 'aʊ', 's'],
            'friend': ['f', 'r', 'e', 'n', 'd'],
            'family': ['f', 'æ', 'm', 'ɪ', 'l', 'ɪ'],
            'book': ['b', 'ʊ', 'k'],
            'music': ['m', 'j', 'uː', 'z', 'ɪ', 'k'],
            'beautiful': ['b', 'j', 'uː', 't', 'ɪ', 'f', 'ʊ', 'l'],
            'important': ['ɪ', 'm', 'p', 'ɔː', 't', 'ə', 'n', 't'],
            'different': ['d', 'ɪ', 'f', 'ər', 'ə', 'n', 't'],
            'because': ['b', 'ɪ', 'k', 'ɒ', 'z']
        }
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if word_clean in phoneme_rules:
                word_phoneme_mapping[word_clean] = phoneme_rules[word_clean]
            else:
                # 对未知单词使用简单的字母到音素映射
                phonemes = []
                for char in word_clean:
                    if char in self.phoneme_map:
                        phonemes.append(self.phoneme_map[char])
                    else:
                        phonemes.append(char)
                word_phoneme_mapping[word_clean] = phonemes
        
        return word_phoneme_mapping
    
    def analyze_word_pronunciation(self, words: List[str], word_phoneme_mapping: Dict[str, List[str]], 
                                 phoneme_scores: List[PhonemeScore]) -> List[Dict]:
        """分析单词级发音质量"""
        word_scores = []
        
        # 创建音素到评分的映射
        phoneme_score_map = {}
        for ps in phoneme_scores:
            if ps.phoneme not in phoneme_score_map:
                phoneme_score_map[ps.phoneme] = []
            phoneme_score_map[ps.phoneme].append(ps)
        
        # 跟踪当前音素位置
        current_phoneme_index = 0
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            
            if word_clean in word_phoneme_mapping:
                word_phonemes = word_phoneme_mapping[word_clean]
                word_phoneme_scores = []
                word_issues = []
                
                # 收集该单词对应的音素评分
                for phoneme in word_phonemes:
                    if current_phoneme_index < len(phoneme_scores):
                        # 寻找匹配的音素评分
                        found_score = None
                        
                        # 首先尝试精确匹配当前位置
                        if (current_phoneme_index < len(phoneme_scores) and 
                            phoneme_scores[current_phoneme_index].phoneme == phoneme):
                            found_score = phoneme_scores[current_phoneme_index]
                            current_phoneme_index += 1
                        else:
                            # 在附近位置搜索匹配的音素
                            for i in range(max(0, current_phoneme_index - 2), 
                                         min(len(phoneme_scores), current_phoneme_index + 3)):
                                if phoneme_scores[i].phoneme == phoneme:
                                    found_score = phoneme_scores[i]
                                    current_phoneme_index = i + 1
                                    break
                        
                        if found_score:
                            word_phoneme_scores.append(found_score)
                            word_issues.extend(found_score.issues)
                
                # 计算单词评分
                if word_phoneme_scores:
                    # 使用加权平均，质量差的音素权重更高（影响更大）
                    scores = [ps.score for ps in word_phoneme_scores]
                    qualities = [ps.quality for ps in word_phoneme_scores]
                    
                    # 质量权重：差的音素对整体影响更大
                    quality_weights = {'excellent': 1.0, 'good': 1.1, 'fair': 1.3, 'poor': 1.5}
                    weighted_scores = []
                    
                    for ps in word_phoneme_scores:
                        weight = quality_weights.get(ps.quality, 1.0)
                        weighted_scores.append(ps.score / weight)  # 质量越差，分数影响越大
                    
                    word_score = np.mean(weighted_scores)
                    
                    # 根据问题数量进一步调整
                    severe_issues = [issue for issue in word_issues if 
                                   ('过短' in issue or '过长' in issue or '能量不足' in issue or '不稳定' in issue)]
                    if len(severe_issues) > len(word_phonemes) * 0.3:
                        word_score *= 0.85  # 严重问题较多时降分
                    
                    # 确定单词质量等级
                    word_quality = self.get_quality_level(word_score)
                    
                    # 生成单词级建议
                    word_suggestions = self._generate_word_suggestions(word_clean, word_phoneme_scores, word_issues)
                    
                    word_analysis = {
                        'word': word_clean,
                        'score': round(word_score, 1),
                        'quality': word_quality,
                        'phonemes': word_phonemes,
                        'phoneme_scores': [{
                            'phoneme': ps.phoneme,
                            'score': ps.score,
                            'quality': ps.quality,
                            'issues': ps.issues
                        } for ps in word_phoneme_scores],
                        'issues': list(set(word_issues)),
                        'suggestions': word_suggestions,
                        'needs_improvement': word_score < 70 or len(severe_issues) > 0
                    }
                    
                    word_scores.append(word_analysis)
                else:
                    # 无法找到对应的音素评分
                    word_scores.append({
                        'word': word_clean,
                        'score': 0,
                        'quality': 'unknown',
                        'phonemes': word_phonemes,
                        'phoneme_scores': [],
                        'issues': ['无法获取该单词的详细发音分析'],
                        'suggestions': [f"请重点练习单词 '{word_clean}' 的发音"],
                        'needs_improvement': True
                    })
        
        return word_scores
    
    def _generate_word_suggestions(self, word: str, phoneme_scores: List[PhonemeScore], issues: List[str]) -> List[str]:
        """为特定单词生成发音改进建议"""
        suggestions = []
        
        # 分析单词中的问题音素
        poor_phonemes = [ps.phoneme for ps in phoneme_scores if ps.quality == 'poor']
        fair_phonemes = [ps.phoneme for ps in phoneme_scores if ps.quality == 'fair']
        
        # 时长问题
        timing_issues = [issue for issue in issues if ('过短' in issue or '过长' in issue)]
        if timing_issues:
            suggestions.append(f"单词 '{word}' 的发音节奏需要调整，注意每个音素的时长")
        
        # 清晰度问题
        clarity_issues = [issue for issue in issues if ('不稳定' in issue or '清晰' in issue or '能量不足' in issue)]
        if clarity_issues:
            suggestions.append(f"单词 '{word}' 需要更清晰的发音，注意口型和发声")
        
        # 特定音素问题
        if poor_phonemes:
            problematic_phonemes = ', '.join(list(set(poor_phonemes)))
            suggestions.append(f"单词 '{word}' 中的音素 [{problematic_phonemes}] 需要重点练习")
        
        if fair_phonemes and not poor_phonemes:  # 只有一般问题时
            suggestions.append(f"单词 '{word}' 的发音基本正确，可以进一步完善")
        
        # 单词特定的发音建议
        word_specific_suggestions = self._get_word_specific_suggestions(word, poor_phonemes + fair_phonemes)
        suggestions.extend(word_specific_suggestions)
        
        # 如果没有问题，给出鼓励
        if not poor_phonemes and not fair_phonemes and not timing_issues and not clarity_issues:
            suggestions.append(f"单词 '{word}' 的发音很好！")
        
        return suggestions
    
    def _get_word_specific_suggestions(self, word: str, problematic_phonemes: List[str]) -> List[str]:
        """获取特定单词的发音技巧建议"""
        suggestions = []
        
        # 常见单词的特殊发音建议
        word_tips = {
            'the': "注意 'th' 的发音，舌尖轻触上齿",
            'this': "'th' 和 'is' 要连接自然，不要停顿",
            'that': "强调 'th' 的摩擦音和 'a' 的开口度",
            'think': "'th' 和 'ink' 的连接要流畅",
            'thank': "'th' + 'ank'，注意鼻音 'n'",
            'with': "'w' 要圆唇，'th' 要清晰",
            'water': "注意 'wa' 的双唇收缩和 'ter' 的卷舌音",
            'work': "'w' 圆唇开始，'or' 要有适当的元音长度",
            'world': "'w' + 'or' + 'ld'，最后的 'ld' 要清晰",
            'beautiful': "注意重音在 'beau'，'ti' 要轻读",
            'important': "重音在 'por'，注意每个音节的清晰度",
            'different': "'dif' + 'fer' + 'ent'，注意重音分配",
            'school': "'sch' 要连续发音，'ool' 要拖长",
            'people': "'peo' + 'ple'，注意最后的轻音节",
            'because': "'be' + 'cause'，重音在 'cause'"
        }
        
        if word in word_tips:
            suggestions.append(word_tips[word])
        
        # 根据问题音素给出建议
        for phoneme in problematic_phonemes:
            if phoneme == 'θ':  # th音
                suggestions.append("练习 'th' 音：舌尖轻触上齿，气流从舌齿间通过")
            elif phoneme == 'ð':  # 浊th音
                suggestions.append("浊 'th' 音要有声带振动，与清 'th' 区分")
            elif phoneme == 'r':
                suggestions.append("英语 'r' 音：舌尖不触碰任何部位，轻微卷起")
            elif phoneme == 'l':
                suggestions.append("'l' 音：舌尖抵住上齿龈，气流从舌侧流出")
            elif phoneme in ['æ', 'ɑː', 'ɒ']:
                suggestions.append("开口元音：需要更大的张口度和更低的舌位")
            elif phoneme in ['iː', 'ɪ']:
                suggestions.append("高元音：舌位要高，嘴型略扁")
            elif phoneme in ['s', 'z']:
                suggestions.append("'s/z' 音：舌尖接近上齿龈，保持狭窄缝隙")
            elif phoneme in ['ʃ', 'ʒ']:
                suggestions.append("'sh/zh' 音：舌身抬高，唇部略圆")
        
        return suggestions