import pandas as pd
import random
import json
import os
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime


class CustomExerciseManager:
    """自定义练习管理器"""
    
    def __init__(self, data_dir="data/custom_exercises"):
        self.data_dir = data_dir
        self.exercises_file = os.path.join(data_dir, "exercises.json")
        self.progress_file = os.path.join(data_dir, "progress.json")
        self._ensure_data_dir()
        self.exercises = self._load_exercises()
        self.progress = self._load_progress()
    
    def _ensure_data_dir(self):
        """确保数据目录存在"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _load_exercises(self) -> Dict[str, Any]:
        """加载练习数据"""
        if os.path.exists(self.exercises_file):
            try:
                with open(self.exercises_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载练习数据失败: {e}")
        return {"exercise_sets": {}}
    
    def _load_progress(self) -> Dict[str, Any]:
        """加载进度数据"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载进度数据失败: {e}")
        return {"user_progress": {}}
    
    def _save_exercises(self):
        """保存练习数据"""
        try:
            with open(self.exercises_file, 'w', encoding='utf-8') as f:
                json.dump(self.exercises, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存练习数据失败: {e}")
    
    def _save_progress(self):
        """保存进度数据"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存进度数据失败: {e}")
    
    def create_exercise_set(self, name: str, description: str = "", exercise_type: str = "mixed") -> str:
        """创建新的练习集
        
        Args:
            name: 练习集名称
            description: 练习集描述
            exercise_type: 练习类型 (speech, phoneme, grammar, vocabulary, listening, mixed)
        
        Returns:
            练习集ID
        """
        exercise_id = str(uuid.uuid4())
        self.exercises["exercise_sets"][exercise_id] = {
            "id": exercise_id,
            "name": name,
            "description": description,
            "type": exercise_type,
            "created_at": datetime.now().isoformat(),
            "items": [],
            "stats": {
                "total_items": 0,
                "completed_count": 0,
                "average_score": 0
            }
        }
        self._save_exercises()
        return exercise_id
    
    def add_exercise_items(self, exercise_id: str, items: List[Dict[str, Any]]) -> bool:
        """添加练习项目到练习集
        
        Args:
            exercise_id: 练习集ID
            items: 练习项目列表
                每个项目应包含: {
                    "type": "speech|phoneme|grammar|vocabulary|listening",
                    "content": {...},  # 练习内容
                    "difficulty": "easy|medium|hard",
                    "tags": [...]  # 标签列表
                }
        """
        if exercise_id not in self.exercises["exercise_sets"]:
            return False
        
        exercise_set = self.exercises["exercise_sets"][exercise_id]
        for item in items:
            item_id = str(uuid.uuid4())
            item["id"] = item_id
            item["created_at"] = datetime.now().isoformat()
            exercise_set["items"].append(item)
        
        exercise_set["stats"]["total_items"] = len(exercise_set["items"])
        self._save_exercises()
        return True
    
    def get_exercise_sets(self) -> List[Dict[str, Any]]:
        """获取所有练习集"""
        return list(self.exercises["exercise_sets"].values())
    
    def get_exercise_set(self, exercise_id: str, user_id: str = "default") -> Optional[Dict[str, Any]]:
        """获取指定练习集，包含用户进度统计"""
        exercise_set = self.exercises["exercise_sets"].get(exercise_id)
        if not exercise_set:
            return None
        
        # 复制一份避免修改原数据
        result = exercise_set.copy()
        result["stats"] = exercise_set["stats"].copy()
        
        # 获取用户进度并更新统计信息
        user_progress = self.get_user_progress(user_id)
        if exercise_id in user_progress:
            exercise_progress = user_progress[exercise_id]
            
            # 计算实际完成的不同题目数（去重）
            completed_item_ids = set(result["item_id"] for result in exercise_progress["results"])
            completed_count = len(completed_item_ids)
            
            # 更新统计信息
            result["stats"]["completed_count"] = completed_count
            result["stats"]["total_attempts"] = exercise_progress["stats"]["total_attempts"]
            result["stats"]["average_score"] = exercise_progress["stats"]["average_score"]
            result["stats"]["best_score"] = exercise_progress["stats"]["best_score"]
            result["stats"]["total_time"] = exercise_progress["stats"]["total_time"]
            
            # 计算完成百分比（基于实际完成的不同题目数）
            total_items = result["stats"]["total_items"]
            if total_items > 0:
                progress_percentage = round((completed_count / total_items * 100), 1)
                # 限制最大100%
                result["stats"]["progress_percentage"] = min(progress_percentage, 100.0)
            else:
                result["stats"]["progress_percentage"] = 0.0
        
        return result
    
    def get_random_exercise_item(self, exercise_id: str, difficulty: str = None, 
                                exercise_type: str = None, user_id: str = "default") -> Optional[Dict[str, Any]]:
        """从练习集中随机获取一个练习项目"""
        exercise_set = self.exercises["exercise_sets"].get(exercise_id)
        if not exercise_set or not exercise_set["items"]:
            return None
        
        items = exercise_set["items"]
        
        # 按难度过滤
        if difficulty:
            items = [item for item in items if item.get("difficulty") == difficulty]
        
        # 按类型过滤
        if exercise_type:
            items = [item for item in items if item.get("type") == exercise_type]
        
        if not items:
            print(f"警告: 未找到符合条件的练习项目 - exercise_id: {exercise_id}, difficulty: {difficulty}, type: {exercise_type}")
            print(f"练习集中的项目类型: {set(item.get('type') for item in exercise_set['items'])}")
            print(f"练习集中的难度: {set(item.get('difficulty') for item in exercise_set['items'])}")
            return None
        
        # 获取用户进度，优先选择未完成的项目
        user_progress = self.get_user_progress(user_id)
        completed_item_ids = set()
        
        if exercise_id in user_progress:
            completed_item_ids = {result["item_id"] for result in user_progress[exercise_id]["results"]}
        
        # 优先选择未完成的项目
        uncompleted_items = [item for item in items if item["id"] not in completed_item_ids]
        
        if uncompleted_items:
            return random.choice(uncompleted_items)
        else:
            # 如果所有项目都完成了，随机选择一个
            return random.choice(items)
    
    def record_exercise_result(self, exercise_id: str, item_id: str, score: float, 
                             time_spent: float, user_id: str = "default") -> bool:
        """记录练习结果"""
        if user_id not in self.progress["user_progress"]:
            self.progress["user_progress"][user_id] = {}
        
        user_progress = self.progress["user_progress"][user_id]
        if exercise_id not in user_progress:
            user_progress[exercise_id] = {
                "exercise_id": exercise_id,
                "results": [],
                "stats": {
                    "total_attempts": 0,
                    "average_score": 0,
                    "best_score": 0,
                    "total_time": 0
                }
            }
        
        exercise_progress = user_progress[exercise_id]
        result = {
            "item_id": item_id,
            "score": score,
            "time_spent": time_spent,
            "timestamp": datetime.now().isoformat()
        }
        
        exercise_progress["results"].append(result)
        
        # 更新统计信息
        stats = exercise_progress["stats"]
        stats["total_attempts"] += 1
        stats["total_time"] += time_spent
        stats["best_score"] = max(stats["best_score"], score)
        
        # 计算平均分
        all_scores = [r["score"] for r in exercise_progress["results"]]
        stats["average_score"] = sum(all_scores) / len(all_scores)
        
        self._save_progress()
        return True
    
    def get_user_progress(self, user_id: str = "default") -> Dict[str, Any]:
        """获取用户进度"""
        return self.progress["user_progress"].get(user_id, {})
    
    def import_from_file(self, file_path: str, exercise_name: str = None) -> str:
        """从文件导入练习数据"""
        if not exercise_name:
            exercise_name = f"导入练习_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 根据文件类型解析数据
        items = []
        
        if file_path.endswith('.csv'):
            items = self._parse_csv_file(file_path)
        elif file_path.endswith('.json'):
            items = self._parse_json_file(file_path)
        elif file_path.endswith('.txt'):
            items = self._parse_txt_file(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        # 创建练习集并添加项目
        exercise_id = self.create_exercise_set(exercise_name, f"从文件 {file_path} 导入")
        self.add_exercise_items(exercise_id, items)
        
        return exercise_id
    
    def _parse_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """解析CSV文件"""
        try:
            df = pd.read_csv(file_path)
            items = []
            
            for _, row in df.iterrows():
                if 'sentence' in df.columns:
                    # 语音练习格式
                    items.append({
                        "type": "speech",
                        "content": {
                            "text": row['sentence'],
                            "audio_path": row.get('audio_path', ''),
                            "phonetic": row.get('phonetic', '')
                        },
                        "difficulty": row.get('difficulty', 'medium'),
                        "tags": row.get('tags', '').split(',') if row.get('tags') else []
                    })
                elif 'chinese' in df.columns and 'english' in df.columns:
                    # 翻译练习格式
                    items.append({
                        "type": "grammar",
                        "content": {
                            "chinese": row['chinese'],
                            "english": row['english'],
                            "explanation": row.get('explanation', '')
                        },
                        "difficulty": row.get('difficulty', 'medium'),
                        "tags": row.get('tags', '').split(',') if row.get('tags') else []
                    })
            
            return items
        except Exception as e:
            print(f"解析CSV文件失败: {e}")
            return []
    
    def _parse_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """解析JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'items' in data:
                return data['items']
            else:
                return []
        except Exception as e:
            print(f"解析JSON文件失败: {e}")
            return []
    
    def _parse_txt_file(self, file_path: str) -> List[Dict[str, Any]]:
        """解析TXT文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            items = []
            for line in lines:
                # 检查是否包含中英文对照（用|或\t分隔）
                if '|' in line:
                    parts = line.split('|', 1)
                    items.append({
                        "type": "grammar",
                        "content": {
                            "chinese": parts[0].strip(),
                            "english": parts[1].strip() if len(parts) > 1 else ""
                        },
                        "difficulty": "medium",
                        "tags": []
                    })
                elif '\t' in line:
                    parts = line.split('\t', 1)
                    items.append({
                        "type": "grammar",
                        "content": {
                            "chinese": parts[0].strip(),
                            "english": parts[1].strip() if len(parts) > 1 else ""
                        },
                        "difficulty": "medium",
                        "tags": []
                    })
                else:
                    # 纯英文句子，用于语音练习
                    items.append({
                        "type": "speech",
                        "content": {
                            "text": line,
                            "audio_path": "",
                            "phonetic": ""
                        },
                        "difficulty": "medium",
                        "tags": []
                    })
                    # 也可以创建音素级练习
                    items.append({
                        "type": "phoneme",
                        "content": {
                            "text": line,
                            "audio_path": "",
                            "phonetic": ""
                        },
                        "difficulty": "medium",
                        "tags": ["phoneme_analysis"]
                    })
            
            return items
        except Exception as e:
            print(f"解析TXT文件失败: {e}")
            return []


# 全局练习管理器实例
_exercise_manager = None

def get_exercise_manager() -> CustomExerciseManager:
    """获取全局练习管理器实例"""
    global _exercise_manager
    if _exercise_manager is None:
        _exercise_manager = CustomExerciseManager()
    return _exercise_manager


# 兼容性函数 - 保持原有API
def load_custom_data(file_path):
    """加载自定义练习数据（兼容性函数）"""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({"sentence": lines})
    return df.to_dict(orient="records")

def get_random_custom_sentence(data_records):
    """从自定义数据中随机选择一条句子（兼容性函数）"""
    return random.choice(data_records)


# 新增便捷函数
def create_text_exercise(content: str, exercise_name: str = None) -> str:
    """从文本内容创建练习"""
    manager = get_exercise_manager()
    
    if not exercise_name:
        exercise_name = f"文本练习_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exercise_id = manager.create_exercise_set(exercise_name, "从文本创建")
    
    # 解析文本内容
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    items = []
    
    for line in lines:
        if '|' in line or '\t' in line:
            # 中英文对照
            separator = '|' if '|' in line else '\t'
            parts = line.split(separator, 1)
            items.append({
                "type": "grammar",
                "content": {
                    "chinese": parts[0].strip(),
                    "english": parts[1].strip() if len(parts) > 1 else ""
                },
                "difficulty": "medium",
                "tags": []
            })
        else:
            # 纯英文句子
            items.append({
                "type": "speech",
                "content": {
                    "text": line,
                    "audio_path": "",
                    "phonetic": ""
                },
                "difficulty": "medium",
                "tags": []
            })
            # 也创建音素级练习选项
            items.append({
                "type": "phoneme",
                "content": {
                    "text": line,
                    "audio_path": "",
                    "phonetic": ""
                },
                "difficulty": "medium",
                "tags": ["phoneme_analysis"]
            })
    
    manager.add_exercise_items(exercise_id, items)
    return exercise_id

def get_exercise_by_type(exercise_id: str, exercise_type: str, difficulty: str = None, user_id: str = "default") -> Optional[Dict[str, Any]]:
    """根据类型获取练习项目"""
    manager = get_exercise_manager()
    return manager.get_random_exercise_item(exercise_id, difficulty, exercise_type, user_id)