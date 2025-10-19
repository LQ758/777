#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库配置和连接管理
"""

import os
import yaml
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from flask import Flask
from src.core.models import db

class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self, config_file="config/config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.db_config = self.config.get('database', {})
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def get_database_uri(self):
        """获取数据库连接URI"""
        db_type = self.db_config.get('type', 'mysql')
        
        if db_type == 'mysql':
            host = self.db_config.get('host', 'localhost')
            port = self.db_config.get('port', 3306)
            database = self.db_config.get('database', 'english_assistant')
            username = self.db_config.get('username', 'root')
            password = self.db_config.get('password', '')
            charset = self.db_config.get('charset', 'utf8mb4')
            
            # 使用PyMySQL作为MySQL驱动
            uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset={charset}"
            return uri
            
        elif db_type == 'sqlite':
            db_path = self.db_config.get('path', 'data/english_assistant.db')
            # 确保SQLite数据库目录存在
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite:///{db_path}"
        
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")
    
    def get_engine_config(self):
        """获取数据库引擎配置"""
        return {
            'pool_size': self.db_config.get('pool_size', 10),
            'pool_recycle': self.db_config.get('pool_recycle', 3600),
            'pool_pre_ping': True,
            'echo': False  # 生产环境设为False
        }

def init_database(app: Flask):
    """初始化数据库"""
    # 加载配置
    db_config = DatabaseConfig()
    database_uri = db_config.get_database_uri()
    engine_config = db_config.get_engine_config()
    
    # 配置Flask-SQLAlchemy
    app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = engine_config
    
    # 初始化数据库
    db.init_app(app)
    
    return db

def create_tables(app: Flask):
    """创建数据表"""
    with app.app_context():
        try:
            # 创建所有表
            db.create_all()
            print("✅ 数据表创建成功")
            
            # 创建默认用户设置
            _create_default_data()
            
        except Exception as e:
            print(f"❌ 数据表创建失败: {e}")
            raise e

def _create_default_data():
    """创建默认数据（如果需要）"""
    try:
        # 检查是否需要创建默认数据
        from src.core.models import User
        
        # 如果没有用户，可以创建一个默认管理员用户
        if User.query.count() == 0:
            print("📝 检测到空数据库，准备创建默认数据...")
            # 这里可以根据需要创建默认数据
            
        db.session.commit()
        
    except Exception as e:
        print(f"创建默认数据失败: {e}")
        db.session.rollback()

def test_database_connection():
    """测试数据库连接"""
    try:
        db_config = DatabaseConfig()
        database_uri = db_config.get_database_uri()
        engine_config = db_config.get_engine_config()
        
        # 创建测试引擎
        engine = create_engine(database_uri, **engine_config)
        
        # 测试连接
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            print("✅ 数据库连接测试成功")
            return True
            
    except Exception as e:
        print(f"❌ 数据库连接测试失败: {e}")
        return False

if __name__ == "__main__":
    # 测试数据库连接
    test_database_connection()