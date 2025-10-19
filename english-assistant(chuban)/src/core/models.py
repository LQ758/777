#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模型定义
定义所有数据表的结构和关系
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy import Text, JSON

db = SQLAlchemy()

class User(db.Model):
    """用户表"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False, index=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    avatar_url = db.Column(db.String(255))
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)
    
    # 关联关系
    learning_records = db.relationship('LearningRecord', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    exercise_sets = db.relationship('ExerciseSet', backref='creator', lazy='dynamic')
    user_progress = db.relationship('UserProgress', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    user_settings = db.relationship('UserSettings', backref='user', uselist=False, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'phone': self.phone,
            'avatar_url': self.avatar_url,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class UserSettings(db.Model):
    """用户设置表"""
    __tablename__ = 'user_settings'
    
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    preferred_difficulty = db.Column(db.String(20), default='medium')
    audio_quality = db.Column(db.String(20), default='standard')
    enable_detailed_feedback = db.Column(db.Boolean, default=True)
    language_preference = db.Column(db.String(10), default='zh-CN')
    notification_enabled = db.Column(db.Boolean, default=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'preferred_difficulty': self.preferred_difficulty,
            'audio_quality': self.audio_quality,
            'enable_detailed_feedback': self.enable_detailed_feedback,
            'language_preference': self.language_preference,
            'notification_enabled': self.notification_enabled,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class ExerciseSet(db.Model):
    """练习集表"""
    __tablename__ = 'exercise_sets'
    
    id = db.Column(db.String(36), primary_key=True)  # UUID
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(Text)
    type = db.Column(db.String(50), nullable=False)  # 'speech', 'grammar', 'phoneme', 'mixed'
    creator_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    is_public = db.Column(db.Boolean, default=False)
    total_items = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # 关联关系
    exercise_items = db.relationship('ExerciseItem', backref='exercise_set', lazy='dynamic', cascade='all, delete-orphan')
    user_progress = db.relationship('UserProgress', backref='exercise_set', lazy='dynamic')
    
    def __repr__(self):
        return f'<ExerciseSet {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'type': self.type,
            'creator_id': self.creator_id,
            'is_public': self.is_public,
            'total_items': self.total_items,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ExerciseItem(db.Model):
    """练习项目表"""
    __tablename__ = 'exercise_items'
    
    id = db.Column(db.String(36), primary_key=True)  # UUID
    exercise_set_id = db.Column(db.String(36), db.ForeignKey('exercise_sets.id'), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # 'speech', 'grammar', 'phoneme', 'vocabulary'
    content = db.Column(JSON, nullable=False)  # 练习内容JSON
    difficulty = db.Column(db.String(20), default='medium')
    tags = db.Column(db.String(500))  # 逗号分隔的标签
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<ExerciseItem {self.id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'exercise_set_id': self.exercise_set_id,
            'type': self.type,
            'content': self.content,
            'difficulty': self.difficulty,
            'tags': self.tags.split(',') if self.tags else [],
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class LearningRecord(db.Model):
    """学习记录表"""
    __tablename__ = 'learning_records'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    exercise_type = db.Column(db.String(50), nullable=False, index=True)  # 'speech', 'grammar', 'phoneme'
    exercise_set_id = db.Column(db.String(36), db.ForeignKey('exercise_sets.id'))
    exercise_item_id = db.Column(db.String(36), db.ForeignKey('exercise_items.id'))
    content = db.Column(Text, nullable=False)  # 练习的原始内容
    user_input = db.Column(Text)  # 用户的输入/录音转文字
    score = db.Column(db.Float, nullable=False)  # 评分结果
    detailed_result = db.Column(JSON)  # 详细分析结果JSON
    practice_time = db.Column(db.Integer, default=0)  # 练习耗时(秒)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f'<LearningRecord {self.id}: {self.exercise_type}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'exercise_type': self.exercise_type,
            'exercise_set_id': self.exercise_set_id,
            'exercise_item_id': self.exercise_item_id,
            'content': self.content,
            'user_input': self.user_input,
            'score': self.score,
            'detailed_result': self.detailed_result,
            'practice_time': self.practice_time,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class UserProgress(db.Model):
    """用户练习进度表"""
    __tablename__ = 'user_progress'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    exercise_set_id = db.Column(db.String(36), db.ForeignKey('exercise_sets.id'))
    exercise_type = db.Column(db.String(50), nullable=False)  # 练习类型统计
    total_attempts = db.Column(db.Integer, default=0)
    best_score = db.Column(db.Float, default=0)
    average_score = db.Column(db.Float, default=0)
    total_time = db.Column(db.Integer, default=0)  # 总练习时间(秒)
    completion_rate = db.Column(db.Float, default=0)  # 完成率
    last_practice = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 设置复合唯一索引
    __table_args__ = (
        db.UniqueConstraint('user_id', 'exercise_set_id', 'exercise_type', name='_user_exercise_type_uc'),
    )
    
    def __repr__(self):
        return f'<UserProgress {self.id}: User{self.user_id}-{self.exercise_type}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'exercise_set_id': self.exercise_set_id,
            'exercise_type': self.exercise_type,
            'total_attempts': self.total_attempts,
            'best_score': self.best_score,
            'average_score': self.average_score,
            'total_time': self.total_time,
            'completion_rate': self.completion_rate,
            'last_practice': self.last_practice.isoformat() if self.last_practice else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }