#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于数据库的用户管理模块
替代JSON文件存储，使用MySQL数据库
"""

import hashlib
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from .models import db, User, UserSettings

class DatabaseUserManager:
    """基于数据库的用户管理器"""
    
    def __init__(self, secret_key="english_assistant_secret_key_2024"):
        self.secret_key = secret_key
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_token(self, user_id: str) -> str:
        """生成JWT令牌"""
        payload = {
            'user_id': str(user_id),
            'exp': datetime.utcnow() + timedelta(days=7),  # 7天过期
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def _verify_token(self, token: str) -> Optional[str]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload['user_id']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def register_user(self, username: str, password: str, email: str, 
                     full_name: str = "") -> Dict[str, Any]:
        """用户注册"""
        try:
            # 验证输入
            if not username or not password or not email:
                return {"success": False, "message": "用户名、密码和邮箱不能为空"}
            
            if len(password) < 6:
                return {"success": False, "message": "密码长度至少6位"}
            
            # 检查用户名和邮箱是否已存在
            existing_user = User.query.filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                if existing_user.username == username:
                    return {"success": False, "message": "用户名已存在"}
                else:
                    return {"success": False, "message": "邮箱已被注册"}
            
            # 创建新用户
            user = User(
                username=username,
                password_hash=self._hash_password(password),
                email=email,
                full_name=full_name,
                created_at=datetime.utcnow(),
                is_active=True
            )
            
            db.session.add(user)
            db.session.flush()  # 获取用户ID
            
            # 创建用户默认设置
            user_settings = UserSettings(
                user_id=user.id,
                preferred_difficulty='medium',
                enable_detailed_feedback=True,
                language_preference='zh-CN'
            )
            
            db.session.add(user_settings)
            db.session.commit()
            
            return {
                "success": True, 
                "message": "注册成功",
                "user_id": user.id,
                "username": username
            }
            
        except Exception as e:
            db.session.rollback()
            return {"success": False, "message": f"注册失败: {str(e)}"}
    
    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """用户登录"""
        try:
            if not username or not password:
                return {"success": False, "message": "用户名和密码不能为空"}
            
            # 查找用户（支持用户名或邮箱登录）
            user = User.query.filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user:
                return {"success": False, "message": "用户不存在"}
            
            if not user.is_active:
                return {"success": False, "message": "账户已被禁用"}
            
            # 验证密码
            if user.password_hash != self._hash_password(password):
                return {"success": False, "message": "密码错误"}
            
            # 更新最后登录时间
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            # 生成令牌
            token = self._generate_token(user.id)
            
            return {
                "success": True,
                "message": "登录成功",
                "token": token,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "full_name": user.full_name,
                    "email": user.email
                }
            }
            
        except Exception as e:
            db.session.rollback()
            return {"success": False, "message": f"登录失败: {str(e)}"}
    
    def verify_user(self, token: str) -> Optional[Dict[str, Any]]:
        """验证用户令牌"""
        try:
            user_id = self._verify_token(token)
            if not user_id:
                return None
            
            user = User.query.filter_by(id=int(user_id), is_active=True).first()
            if not user:
                return None
            
            # 获取用户设置
            settings = {}
            if user.user_settings:
                settings = user.user_settings.to_dict()
            
            return {
                "id": user.id,
                "username": user.username,
                "full_name": user.full_name,
                "email": user.email,
                "settings": settings
            }
            
        except Exception as e:
            print(f"验证用户令牌失败: {e}")
            return None
    
    def update_user_settings(self, user_id: str, settings: Dict[str, Any]) -> bool:
        """更新用户设置"""
        try:
            user = User.query.get(int(user_id))
            if not user:
                return False
            
            # 获取或创建用户设置
            user_settings = user.user_settings
            if not user_settings:
                user_settings = UserSettings(user_id=user.id)
                db.session.add(user_settings)
            
            # 更新设置
            for key, value in settings.items():
                if hasattr(user_settings, key):
                    setattr(user_settings, key, value)
            
            user_settings.updated_at = datetime.utcnow()
            db.session.commit()
            return True
            
        except Exception as e:
            db.session.rollback()
            print(f"更新用户设置失败: {e}")
            return False
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取用户信息"""
        try:
            user = User.query.get(int(user_id))
            if not user:
                return None
            
            user_dict = user.to_dict()
            if user.user_settings:
                user_dict['settings'] = user.user_settings.to_dict()
            else:
                user_dict['settings'] = {}
                
            return user_dict
            
        except Exception as e:
            print(f"获取用户信息失败: {e}")
            return None
    
    def logout_user(self, token: str) -> bool:
        """用户登出（在无状态JWT中，主要是客户端删除token）"""
        # JWT是无状态的，服务端无需做特殊处理
        # 在生产环境中，可以维护一个黑名单来撤销token
        return True
    
    def deactivate_user(self, user_id: str) -> bool:
        """停用用户账户"""
        try:
            user = User.query.get(int(user_id))
            if not user:
                return False
            
            user.is_active = False
            db.session.commit()
            return True
            
        except Exception as e:
            db.session.rollback()
            print(f"停用用户失败: {e}")
            return False
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> Dict[str, Any]:
        """修改密码"""
        try:
            if len(new_password) < 6:
                return {"success": False, "message": "新密码长度至少6位"}
            
            user = User.query.get(int(user_id))
            if not user:
                return {"success": False, "message": "用户不存在"}
            
            # 验证旧密码
            if user.password_hash != self._hash_password(old_password):
                return {"success": False, "message": "原密码错误"}
            
            # 更新密码
            user.password_hash = self._hash_password(new_password)
            db.session.commit()
            
            return {"success": True, "message": "密码修改成功"}
            
        except Exception as e:
            db.session.rollback()
            return {"success": False, "message": f"密码修改失败: {str(e)}"}


# 全局用户管理器实例
_db_user_manager = None

def get_db_user_manager() -> DatabaseUserManager:
    """获取全局数据库用户管理器实例"""
    global _db_user_manager
    if _db_user_manager is None:
        _db_user_manager = DatabaseUserManager()
    return _db_user_manager