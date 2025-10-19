#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于数据库的学习记录管理模块
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import desc, and_, func
from src.core.models import db, LearningRecord, UserProgress, User, ExerciseSet, ExerciseItem

class DatabaseLearningManager:
    """基于数据库的学习记录管理器"""
    
    def add_learning_record(self, user_id: int, exercise_type: str, 
                           content: str, user_input: str, score: float,
                           detailed_result: Dict[str, Any] = None,
                           practice_time: int = 0,
                           exercise_set_id: str = None,
                           exercise_item_id: str = None) -> Optional[int]:
        """添加学习记录"""
        try:
            record = LearningRecord(
                user_id=user_id,
                exercise_type=exercise_type,
                exercise_set_id=exercise_set_id,
                exercise_item_id=exercise_item_id,
                content=content,
                user_input=user_input,
                score=score,
                detailed_result=detailed_result or {},
                practice_time=practice_time,
                created_at=datetime.utcnow()
            )
            
            db.session.add(record)
            db.session.flush()  # 获取记录ID
            
            # 更新用户进度
            self._update_user_progress(user_id, exercise_type, score, practice_time, exercise_set_id)
            
            db.session.commit()
            return record.id
            
        except Exception as e:
            db.session.rollback()
            print(f"添加学习记录失败: {e}")
            return None
    
    def _update_user_progress(self, user_id: int, exercise_type: str, 
                             score: float, practice_time: int, exercise_set_id: str = None):
        """更新用户进度统计"""
        try:
            # 查找现有进度记录
            progress = UserProgress.query.filter_by(
                user_id=user_id,
                exercise_type=exercise_type,
                exercise_set_id=exercise_set_id
            ).first()
            
            if not progress:
                # 创建新的进度记录
                progress = UserProgress(
                    user_id=user_id,
                    exercise_set_id=exercise_set_id,
                    exercise_type=exercise_type,
                    total_attempts=0,
                    best_score=0,
                    average_score=0,
                    total_time=0,
                    completion_rate=0,
                    created_at=datetime.utcnow()
                )
                db.session.add(progress)
            
            # 更新统计数据
            progress.total_attempts += 1
            progress.best_score = max(progress.best_score, score)
            progress.total_time += practice_time
            progress.last_practice = datetime.utcnow()
            progress.updated_at = datetime.utcnow()
            
            # 重新计算平均分
            total_score = (progress.average_score * (progress.total_attempts - 1)) + score
            progress.average_score = total_score / progress.total_attempts
            
            # 如果有练习集ID，计算完成率
            if exercise_set_id:
                exercise_set = ExerciseSet.query.get(exercise_set_id)
                if exercise_set and exercise_set.total_items > 0:
                    completed_items = db.session.query(func.count(func.distinct(LearningRecord.exercise_item_id))).filter(
                        LearningRecord.user_id == user_id,
                        LearningRecord.exercise_set_id == exercise_set_id
                    ).scalar()
                    progress.completion_rate = (completed_items / exercise_set.total_items) * 100
            
        except Exception as e:
            print(f"更新用户进度失败: {e}")
    
    def get_user_records(self, user_id: int, limit: int = 50, 
                        exercise_type: str = None,
                        start_date: datetime = None,
                        end_date: datetime = None) -> List[Dict[str, Any]]:
        """获取用户学习记录"""
        try:
            query = LearningRecord.query.filter_by(user_id=user_id)
            
            if exercise_type:
                query = query.filter_by(exercise_type=exercise_type)
            
            if start_date:
                query = query.filter(LearningRecord.created_at >= start_date)
            
            if end_date:
                query = query.filter(LearningRecord.created_at <= end_date)
            
            records = query.order_by(desc(LearningRecord.created_at)).limit(limit).all()
            
            return [record.to_dict() for record in records]
            
        except Exception as e:
            print(f"获取用户学习记录失败: {e}")
            return []
    
    def get_user_progress(self, user_id: int, exercise_set_id: str = None) -> List[Dict[str, Any]]:
        """获取用户进度统计"""
        try:
            query = UserProgress.query.filter_by(user_id=user_id)
            
            if exercise_set_id:
                query = query.filter_by(exercise_set_id=exercise_set_id)
            
            progress_records = query.all()
            return [progress.to_dict() for progress in progress_records]
            
        except Exception as e:
            print(f"获取用户进度失败: {e}")
            return []
    
    def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """获取用户统计信息"""
        try:
            # 基础统计
            total_records = LearningRecord.query.filter_by(user_id=user_id).count()
            
            if total_records == 0:
                return {
                    "total_practices": 0,
                    "total_time": 0,
                    "average_score": 0,
                    "best_score": 0,
                    "exercise_types": {},
                    "recent_performance": []
                }
            
            # 总体统计
            stats = db.session.query(
                func.sum(LearningRecord.practice_time).label('total_time'),
                func.avg(LearningRecord.score).label('avg_score'),
                func.max(LearningRecord.score).label('best_score')
            ).filter_by(user_id=user_id).first()
            
            # 按练习类型统计
            type_stats = db.session.query(
                LearningRecord.exercise_type,
                func.count(LearningRecord.id).label('count'),
                func.avg(LearningRecord.score).label('avg_score'),
                func.max(LearningRecord.score).label('best_score'),
                func.sum(LearningRecord.practice_time).label('total_time')
            ).filter_by(user_id=user_id).group_by(LearningRecord.exercise_type).all()
            
            exercise_types = {}
            for stat in type_stats:
                exercise_types[stat.exercise_type] = {
                    'total_attempts': stat.count,
                    'average_score': round(stat.avg_score, 2) if stat.avg_score else 0,
                    'best_score': stat.best_score or 0,
                    'total_time': stat.total_time or 0
                }
            
            # 最近7天的表现趋势
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            recent_performance = db.session.query(
                func.date(LearningRecord.created_at).label('date'),
                func.count(LearningRecord.id).label('practices'),
                func.avg(LearningRecord.score).label('avg_score')
            ).filter(
                LearningRecord.user_id == user_id,
                LearningRecord.created_at >= seven_days_ago
            ).group_by(func.date(LearningRecord.created_at)).order_by('date').all()
            
            recent_data = []
            for perf in recent_performance:
                recent_data.append({
                    'date': perf.date.isoformat(),
                    'practices': perf.practices,
                    'average_score': round(perf.avg_score, 2) if perf.avg_score else 0
                })
            
            return {
                "total_practices": total_records,
                "total_time": int(stats.total_time or 0),
                "average_score": round(stats.avg_score, 2) if stats.avg_score else 0,
                "best_score": stats.best_score or 0,
                "exercise_types": exercise_types,
                "recent_performance": recent_data
            }
            
        except Exception as e:
            print(f"获取用户统计信息失败: {e}")
            return {}
    
    def get_recent_activity(self, user_id: int, days: int = 7) -> List[Dict[str, Any]]:
        """获取用户最近活动"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            records = LearningRecord.query.filter(
                LearningRecord.user_id == user_id,
                LearningRecord.created_at >= cutoff_date
            ).order_by(desc(LearningRecord.created_at)).all()
            
            return [record.to_dict() for record in records]
            
        except Exception as e:
            print(f"获取最近活动失败: {e}")
            return []
    
    def get_learning_streaks(self, user_id: int) -> Dict[str, Any]:
        """获取学习连续记录"""
        try:
            # 获取最近30天的学习日期
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            practice_dates = db.session.query(
                func.date(LearningRecord.created_at).label('practice_date')
            ).filter(
                LearningRecord.user_id == user_id,
                LearningRecord.created_at >= thirty_days_ago
            ).group_by(func.date(LearningRecord.created_at)).order_by('practice_date').all()
            
            if not practice_dates:
                return {"current_streak": 0, "longest_streak": 0, "total_days": 0}
            
            dates = [pd.practice_date for pd in practice_dates]
            
            # 计算当前连续天数
            current_streak = 0
            today = datetime.utcnow().date()
            check_date = today
            
            while check_date in dates:
                current_streak += 1
                check_date = check_date - timedelta(days=1)
            
            # 计算最长连续天数
            longest_streak = 0
            current_temp_streak = 1
            
            for i in range(1, len(dates)):
                if (dates[i] - dates[i-1]).days == 1:
                    current_temp_streak += 1
                else:
                    longest_streak = max(longest_streak, current_temp_streak)
                    current_temp_streak = 1
            
            longest_streak = max(longest_streak, current_temp_streak)
            
            return {
                "current_streak": current_streak,
                "longest_streak": longest_streak,
                "total_days": len(dates)
            }
            
        except Exception as e:
            print(f"获取学习连续记录失败: {e}")
            return {"current_streak": 0, "longest_streak": 0, "total_days": 0}
    
    def get_improvement_analysis(self, user_id: int, exercise_type: str = None, days: int = 30) -> Dict[str, Any]:
        """获取学习改进分析"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = LearningRecord.query.filter(
                LearningRecord.user_id == user_id,
                LearningRecord.created_at >= cutoff_date
            )
            
            if exercise_type:
                query = query.filter_by(exercise_type=exercise_type)
            
            records = query.order_by(LearningRecord.created_at).all()
            
            if len(records) < 2:
                return {"improvement": 0, "trend": "insufficient_data", "analysis": "数据不足，无法分析"}
            
            # 计算前半段和后半段的平均分
            mid_point = len(records) // 2
            first_half_avg = sum(r.score for r in records[:mid_point]) / mid_point
            second_half_avg = sum(r.score for r in records[mid_point:]) / (len(records) - mid_point)
            
            improvement = second_half_avg - first_half_avg
            
            if improvement > 5:
                trend = "improving"
                analysis = "学习效果显著提升！继续保持这种学习节奏。"
            elif improvement > 1:
                trend = "slightly_improving"
                analysis = "学习效果稳步提升，建议增加练习频率。"
            elif improvement > -1:
                trend = "stable"
                analysis = "学习效果保持稳定，可以尝试挑战更高难度。"
            else:
                trend = "declining"
                analysis = "学习效果有所下降，建议回顾基础知识并调整学习方法。"
            
            return {
                "improvement": round(improvement, 2),
                "trend": trend,
                "analysis": analysis,
                "first_half_avg": round(first_half_avg, 2),
                "second_half_avg": round(second_half_avg, 2),
                "total_practices": len(records)
            }
            
        except Exception as e:
            print(f"获取学习改进分析失败: {e}")
            return {"improvement": 0, "trend": "error", "analysis": "分析失败"}


# 全局学习记录管理器实例
_db_learning_manager = None

def get_db_learning_manager() -> DatabaseLearningManager:
    """获取全局数据库学习记录管理器实例"""
    global _db_learning_manager
    if _db_learning_manager is None:
        _db_learning_manager = DatabaseLearningManager()
    return _db_learning_manager