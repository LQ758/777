import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import scipy.signal
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AcousticFeatureExtractor:
    """声学特征提取器"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.hop_length = 512
        self.n_fft = 2048
        
    def extract_f0_features(self, audio: np.ndarray) -> Dict:
        """提取基频相关特征"""
        try:
            # 使用yin算法提取基频
            f0 = librosa.yin(audio, fmin=80, fmax=400, sr=self.sr)
            
            # 过滤无效值
            valid_f0 = f0[f0 > 0]
            
            if len(valid_f0) == 0:
                return {
                    'f0_mean': 0,
                    'f0_std': 0,
                    'f0_median': 0,
                    'f0_range': 0,
                    'f0_slope': 0,
                    'voicing_rate': 0
                }
            
            features = {
                'f0_mean': np.mean(valid_f0),
                'f0_std': np.std(valid_f0),
                'f0_median': np.median(valid_f0),
                'f0_range': np.max(valid_f0) - np.min(valid_f0),
                'voicing_rate': len(valid_f0) / len(f0)  # 有声段比例
            }
            
            # 计算基频变化趋势
            if len(valid_f0) > 1:
                time_points = np.arange(len(valid_f0))
                slope, _, _, _, _ = stats.linregress(time_points, valid_f0)
                features['f0_slope'] = slope
            else:
                features['f0_slope'] = 0
            
            return features
            
        except Exception as e:
            print(f"基频特征提取失败: {e}")
            return {'f0_mean': 0, 'f0_std': 0, 'f0_median': 0, 'f0_range': 0, 'f0_slope': 0, 'voicing_rate': 0}
    
    def extract_formant_features(self, audio: np.ndarray) -> Dict:
        """提取共振峰特征（简化版本）"""
        try:
            # 使用LPC方法估算共振峰
            # 这是一个简化的实现，实际应用中可能需要更专业的共振峰检测
            
            # 预加重
            pre_emphasis = 0.97
            emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # 计算LPC系数
            lpc_order = 10
            if len(emphasized) > lpc_order:
                # 使用librosa的lpc估算
                lpc_coeffs = librosa.lpc(emphasized, order=lpc_order)
                
                # 从LPC系数计算频率响应
                w, h = scipy.signal.freqz(1, lpc_coeffs, worN=512, fs=self.sr)
                
                # 找到峰值作为共振峰的近似
                peaks, _ = scipy.signal.find_peaks(np.abs(h), height=0.1, distance=10)
                
                formants = w[peaks] if len(peaks) > 0 else []
                
                # 提取前3个共振峰
                f1 = formants[0] if len(formants) > 0 else 0
                f2 = formants[1] if len(formants) > 1 else 0
                f3 = formants[2] if len(formants) > 2 else 0
                
                return {
                    'f1': f1,
                    'f2': f2,
                    'f3': f3,
                    'f1_f2_ratio': f2/f1 if f1 > 0 else 0,
                    'formant_bandwidth': np.std(formants) if len(formants) > 1 else 0
                }
            else:
                return {'f1': 0, 'f2': 0, 'f3': 0, 'f1_f2_ratio': 0, 'formant_bandwidth': 0}
                
        except Exception as e:
            print(f"共振峰特征提取失败: {e}")
            return {'f1': 0, 'f2': 0, 'f3': 0, 'f1_f2_ratio': 0, 'formant_bandwidth': 0}
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """提取频谱特征"""
        try:
            # 频谱质心
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
            
            # 频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
            
            # 频谱对比度
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
            
            # 频谱滚降点
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
            
            # 梅尔频谱系数 (MFCC)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
            
            # 色度特征
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
            
            return {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'mfcc_mean': np.mean(mfccs, axis=1),
                'mfcc_std': np.std(mfccs, axis=1),
                'chroma_mean': np.mean(chroma, axis=1),
                'chroma_std': np.std(chroma, axis=1)
            }
            
        except Exception as e:
            print(f"频谱特征提取失败: {e}")
            return {}
    
    def extract_temporal_features(self, audio: np.ndarray) -> Dict:
        """提取时域特征"""
        try:
            # RMS能量
            rms = librosa.feature.rms(y=audio)[0]
            
            # 零交叉率
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # 短时能量
            frame_length = 1024
            hop_length = 512
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames**2, axis=0)
            
            return {
                'rms_mean': np.mean(rms),
                'rms_std': np.std(rms),
                'zcr_mean': np.mean(zcr),
                'zcr_std': np.std(zcr),
                'energy_mean': np.mean(energy),
                'energy_std': np.std(energy),
                'energy_max': np.max(energy) if len(energy) > 0 else 0,
                'silence_ratio': np.sum(energy < np.mean(energy) * 0.1) / len(energy) if len(energy) > 0 else 0
            }
            
        except Exception as e:
            print(f"时域特征提取失败: {e}")
            return {}
    
    def extract_all_features(self, audio: np.ndarray) -> Dict:
        """提取所有声学特征"""
        if len(audio) == 0:
            return {}
        
        features = {}
        
        # 基频特征
        f0_features = self.extract_f0_features(audio)
        features.update(f0_features)
        
        # 共振峰特征
        formant_features = self.extract_formant_features(audio)
        features.update(formant_features)
        
        # 频谱特征
        spectral_features = self.extract_spectral_features(audio)
        features.update(spectral_features)
        
        # 时域特征
        temporal_features = self.extract_temporal_features(audio)
        features.update(temporal_features)
        
        return features


class PhonemeAligner:
    """音素对齐器"""
    
    def __init__(self):
        self.min_phoneme_duration = 0.03  # 最小音素时长（秒）
        self.max_phoneme_duration = 0.4   # 最大音素时长（秒）
    
    def simple_uniform_alignment(self, audio_length: int, phoneme_sequence: List[str], 
                                sr: int = 16000) -> List[Tuple[str, float, float]]:
        """简单的均匀对齐"""
        total_duration = audio_length / sr
        phoneme_count = len(phoneme_sequence)
        
        if phoneme_count == 0:
            return []
        
        alignments = []
        avg_duration = total_duration / phoneme_count
        
        for i, phoneme in enumerate(phoneme_sequence):
            start_time = i * avg_duration
            end_time = (i + 1) * avg_duration
            alignments.append((phoneme, start_time, end_time))
        
        return alignments
    
    def duration_weighted_alignment(self, audio_length: int, phoneme_sequence: List[str],
                                  sr: int = 16000) -> List[Tuple[str, float, float]]:
        """基于音素典型时长的加权对齐"""
        # 音素典型时长权重
        duration_weights = {
            # 元音通常较长
            'æ': 1.2, 'ɪ': 1.0, 'ʊ': 1.0, 'iː': 1.5, 'uː': 1.5, 'ɜː': 1.8,
            'ʌ': 1.1, 'aɪ': 1.3, 'aʊ': 1.3, 'ɔɪ': 1.3, 'e': 1.1, 'ɒ': 1.1,
            
            # 辅音通常较短
            'p': 0.6, 'b': 0.7, 't': 0.6, 'd': 0.7, 'k': 0.6, 'g': 0.7,
            'f': 0.9, 'v': 0.8, 'θ': 0.8, 'ð': 0.7, 's': 0.9, 'z': 0.8,
            'ʃ': 0.9, 'ʒ': 0.8, 'tʃ': 0.8, 'dʒ': 0.8,
            'm': 0.8, 'n': 0.8, 'ŋ': 0.8, 'l': 0.8, 'r': 0.8,
            'w': 0.7, 'j': 0.6, 'h': 0.5
        }
        
        total_duration = audio_length / sr
        phoneme_count = len(phoneme_sequence)
        
        if phoneme_count == 0:
            return []
        
        # 计算权重总和
        weights = [duration_weights.get(ph, 1.0) for ph in phoneme_sequence]
        total_weight = sum(weights)
        
        # 分配时长
        alignments = []
        current_time = 0.0
        
        for i, (phoneme, weight) in enumerate(zip(phoneme_sequence, weights)):
            duration = (weight / total_weight) * total_duration
            
            # 限制在合理范围内
            duration = max(self.min_phoneme_duration, 
                          min(self.max_phoneme_duration, duration))
            
            start_time = current_time
            end_time = current_time + duration
            
            alignments.append((phoneme, start_time, end_time))
            current_time = end_time
        
        return alignments
    
    def energy_based_alignment(self, audio: np.ndarray, phoneme_sequence: List[str],
                              sr: int = 16000) -> List[Tuple[str, float, float]]:
        """基于能量变化的对齐"""
        try:
            # 计算短时能量
            hop_length = 512
            frame_length = 1024
            
            # 使用RMS作为能量指标
            rms = librosa.feature.rms(y=audio, hop_length=hop_length, frame_length=frame_length)[0]
            
            # 平滑能量曲线
            from scipy.ndimage import gaussian_filter1d
            smoothed_rms = gaussian_filter1d(rms, sigma=2)
            
            # 找到能量变化点
            diff = np.diff(smoothed_rms)
            change_points = np.where(np.abs(diff) > np.std(diff) * 0.5)[0]
            
            # 转换为时间
            time_per_frame = hop_length / sr
            change_times = change_points * time_per_frame
            
            # 根据变化点分割音素
            total_duration = len(audio) / sr
            phoneme_count = len(phoneme_sequence)
            
            if phoneme_count == 0:
                return []
            
            # 如果变化点不够，使用均匀分割
            if len(change_times) < phoneme_count - 1:
                return self.simple_uniform_alignment(len(audio), phoneme_sequence, sr)
            
            # 选择合适的分割点
            segment_times = [0.0]
            if len(change_times) > 0:
                selected_points = np.linspace(0, len(change_times)-1, phoneme_count-1, dtype=int)
                segment_times.extend(change_times[selected_points])
            segment_times.append(total_duration)
            
            alignments = []
            for i in range(phoneme_count):
                start_time = segment_times[i]
                end_time = segment_times[i + 1]
                alignments.append((phoneme_sequence[i], start_time, end_time))
            
            return alignments
            
        except Exception as e:
            print(f"基于能量的对齐失败: {e}")
            return self.simple_uniform_alignment(len(audio), phoneme_sequence, sr)


class PronunciationQualityAssessor:
    """发音质量评估器"""
    
    def __init__(self):
        # 音素类别特征期望值
        self.vowel_expectations = {
            'spectral_centroid': (800, 1500),    # 元音的频谱质心期望范围
            'f1_range': (200, 1000),             # 第一共振峰范围
            'f2_range': (800, 2500),             # 第二共振峰范围
            'voicing_rate': (0.8, 1.0)           # 有声率
        }
        
        self.consonant_expectations = {
            'fricative': {  # 摩擦音 f, v, s, z, ʃ, ʒ, θ, ð
                'spectral_centroid': (2000, 8000),
                'zcr': (0.3, 0.8),
                'energy_stability': 0.3
            },
            'stop': {  # 爆破音 p, b, t, d, k, g
                'energy_burst': True,
                'silence_before': True,
                'short_duration': (0.02, 0.1)
            },
            'nasal': {  # 鼻音 m, n, ŋ
                'f1_range': (200, 400),
                'low_energy': True,
                'voicing_rate': (0.9, 1.0)
            }
        }
    
    def classify_phoneme(self, phoneme: str) -> str:
        """音素分类"""
        vowels = ['æ', 'ɪ', 'ʊ', 'iː', 'uː', 'ɜː', 'ʌ', 'aɪ', 'aʊ', 'ɔɪ', 'e', 'ɒ', 'ɑː']
        fricatives = ['f', 'v', 's', 'z', 'ʃ', 'ʒ', 'θ', 'ð', 'h']
        stops = ['p', 'b', 't', 'd', 'k', 'g']
        nasals = ['m', 'n', 'ŋ']
        liquids = ['l', 'r']
        glides = ['w', 'j']
        
        if phoneme in vowels:
            return 'vowel'
        elif phoneme in fricatives:
            return 'fricative'
        elif phoneme in stops:
            return 'stop'
        elif phoneme in nasals:
            return 'nasal'
        elif phoneme in liquids:
            return 'liquid'
        elif phoneme in glides:
            return 'glide'
        else:
            return 'unknown'
    
    def assess_vowel_quality(self, features: Dict, phoneme: str) -> Tuple[float, List[str]]:
        """评估元音质量"""
        score = 100.0
        issues = []
        
        # 检查有声率
        if 'voicing_rate' in features:
            if features['voicing_rate'] < 0.7:
                score -= 20
                issues.append(f"元音'{phoneme}'有声程度不足")
        
        # 检查频谱质心（音色）
        if 'spectral_centroid_mean' in features:
            centroid = features['spectral_centroid_mean']
            expected_min, expected_max = self.vowel_expectations['spectral_centroid']
            if centroid < expected_min or centroid > expected_max:
                score -= 15
                issues.append(f"元音'{phoneme}'音色偏离标准")
        
        # 检查共振峰
        if 'f1' in features and 'f2' in features:
            f1, f2 = features['f1'], features['f2']
            if f1 > 0 and f2 > 0:
                # 特定元音的共振峰检查
                if phoneme in ['æ', 'ɑː'] and f1 < 600:  # 低元音应该有较高的F1
                    score -= 12
                    issues.append(f"低元音'{phoneme}'开口度不够")
                elif phoneme in ['iː', 'ɪ'] and f1 > 500:  # 高元音应该有较低的F1
                    score -= 12
                    issues.append(f"高元音'{phoneme}'舌位偏低")
        
        return max(0, score), issues
    
    def assess_consonant_quality(self, features: Dict, phoneme: str, phoneme_type: str) -> Tuple[float, List[str]]:
        """评估辅音质量"""
        score = 100.0
        issues = []
        
        if phoneme_type == 'fricative':
            # 摩擦音应该有高频成分
            if 'spectral_centroid_mean' in features:
                if features['spectral_centroid_mean'] < 2000:
                    score -= 20
                    issues.append(f"摩擦音'{phoneme}'高频成分不足")
            
            # 摩擦音应该有较高的零交叉率
            if 'zcr_mean' in features:
                if features['zcr_mean'] < 0.1:
                    score -= 15
                    issues.append(f"摩擦音'{phoneme}'摩擦特征不明显")
        
        elif phoneme_type == 'stop':
            # 爆破音应该有明显的能量突变
            if 'energy_max' in features and 'energy_mean' in features:
                energy_ratio = features['energy_max'] / features['energy_mean'] if features['energy_mean'] > 0 else 0
                if energy_ratio < 3:
                    score -= 18
                    issues.append(f"爆破音'{phoneme}'爆破特征不明显")
        
        elif phoneme_type == 'nasal':
            # 鼻音应该有低频特征和较强的有声性
            if 'spectral_centroid_mean' in features:
                if features['spectral_centroid_mean'] > 1500:
                    score -= 15
                    issues.append(f"鼻音'{phoneme}'频谱过高")
            
            if 'voicing_rate' in features:
                if features['voicing_rate'] < 0.8:
                    score -= 12
                    issues.append(f"鼻音'{phoneme}'有声性不足")
        
        return max(0, score), issues
    
    def assess_phoneme_quality(self, phoneme: str, features: Dict, duration: float) -> Tuple[float, List[str]]:
        """综合评估音素质量"""
        phoneme_type = self.classify_phoneme(phoneme)
        
        if phoneme_type == 'vowel':
            return self.assess_vowel_quality(features, phoneme)
        else:
            return self.assess_consonant_quality(features, phoneme, phoneme_type)