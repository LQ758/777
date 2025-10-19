# 感觉差点什么的语学助手

一个基于AI技术的智能英语学习助手系统，提供发音评分、语法检测、语音转写等功能。

## 📋 项目概述

本项目是一个现代化的Web应用程序，旨在帮助用户提高英语学习能力。系统集成了多种AI技术，包括语音识别、发音评分、语法检测等，为用户提供个性化的学习体验。

### 主要功能

- 🎤 **英语发音评分**：基于Wav2Vec2模型的智能发音评估
- ✏️ **英文语法检测**：使用LanguageTool进行准确的语法错误检测
- 🔊 **语音转写**：Whisper模型实现的高精度语音转文字
- 📚 **自定义练习**：支持用户创建和管理个人学习内容
- 👤 **用户系统**：完整的用户注册、登录和学习记录管理

## 🛠️ 技术栈

### 后端技术
- **Python 3.8+**
- **Flask 3.0.3** - Web框架
- **MySQL 8.0+** - 数据库
- **SQLAlchemy 2.0.25** - ORM框架
- **PyTorch 2.2.2** - 深度学习框架
- **LibROSA 0.10.2** - 音频处理
- **LanguageTool** - 语法检测

### AI模型
- **Wav2Vec2** - 语音识别和发音评分
- **Whisper** - 语音转文字
- **音素级分析** - 详细发音评估

### 前端技术
- **HTML5/CSS3/JavaScript**
- **Tailwind CSS** - UI框架
- **ECharts** - 数据可视化
- **Axios** - HTTP客户端

## 📁 项目结构

```
english-assistant(chuban)/
├── app.py                 # Flask应用主文件
├── main.py               # 命令行版本入口
├── requirements.txt      # Python依赖
├── config/
│   └── config.yaml      # 配置文件
├── src/
│   ├── core/            # 核心功能模块
│   │   ├── 发音评分模块.py
│   │   ├── 语法检查.py
│   │   ├── 语音转写.py
│   │   ├── 自定义练习模块.py
│   │   ├── database.py
│   │   ├── db_user_manager.py
│   │   └── ...
│   └── utils/           # 工具函数
├── templates/           # HTML模板
│   ├── index.html      # 主页面
│   └── login.html      # 登录页面
├── data/               # 数据文件
│   ├── models/         # AI模型文件
│   ├── audio/          # 音频文件
│   └── ...
└── static/             # 静态资源
```

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- MySQL 8.0 或更高版本
- FFmpeg（用于音频处理）
- 4GB+ 内存（AI模型运行需要）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd english-assistant(chuban)
```

2. **安装Python依赖**
```bash
pip install -r requirements.txt
```

3. **创建数据库**
```sql
CREATE DATABASE english_assistant CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

4. **配置数据库连接**
编辑 `config/config.yaml` 文件：
```yaml
database:
  host: "localhost"
  port: 3306
  database: "english_assistant"
  username: "root"
  password: "your_password"
```

5. **下载AI模型**
- 将Wav2Vec2模型文件放置到 `data/models/wav2vec2-base-960h/` 目录
- 系统会自动下载Whisper模型

6. **启动应用**
```bash
python app.py
```

7. **访问应用**
打开浏览器访问 `http://localhost:5000`

## 🎯 核心功能详解

### 1. 英语发音评分

**功能描述**：
- 用户朗读英语句子，系统评估发音准确度
- 提供0-100分的评分结果
- 支持简化模式和详细分析模式

**技术实现**：
- 使用Wav2Vec2模型进行语音识别
- 基于Levenshtein距离计算发音相似度
- 音素级分析提供详细改进建议

**使用方法**：
1. 点击"生成新句子"获取练习内容
2. 点击录音按钮开始录音
3. 选择评分模式（简化/详细）
4. 点击"提交评分"获取结果

### 2. 英文语法检测

**功能描述**：
- 检测英文文本中的语法错误
- 提供详细的错误说明和修改建议
- 支持文本输入和语音输入两种方式

**技术实现**：
- 集成LanguageTool语法检测引擎
- 错误信息中英文对照
- 实时语法分析和建议生成

**使用方法**：
1. 获取中文句子进行翻译
2. 输入或录音英文翻译
3. 点击"提交检测"查看语法分析结果

### 3. 自定义练习

**功能描述**：
- 用户可创建个人练习集
- 支持语音练习和语法练习两种模式
- 提供学习进度跟踪和统计分析

**技术实现**：
- 练习数据存储在数据库中
- 支持文本批量导入
- 自动统计学习进度和平均分数

**使用方法**：
1. 创建新的练习集或选择已有练习集
2. 导入练习内容（文本格式）
3. 开始练习并查看结果统计

## 🔧 配置说明

### config.yaml 主要配置项

```yaml
# 数据库配置
database:
  type: "mysql"
  host: "localhost"
  port: 3306
  database: "english_assistant"
  username: "root"
  password: "your_password"

# AI模型配置
model:
  whisper: "small"
  wav2vec2: "facebook/wav2vec2-base-960h"

# 音素级评分配置
phoneme_scoring:
  enabled: true
  alignment_method: "ctc"
  scoring_weights:
    duration: 0.3
    quality: 0.5
    consistency: 0.2

# 语法检测配置
language_tool:
  server: "https://api.languagetool.org"
  language: "en-US"
  mother_tongue: "zh-CN"

# 音频配置
audio:
  duration: 5
  sample_rate: 16000
  channels: 1
  format: "int16"
```

## 📊 API接口

### 主要API端点

#### 用户认证
- `POST /api/login` - 用户登录
- `POST /api/logout` - 用户登出

#### 发音评分
- `POST /api/score-pronunciation` - 标准发音评分
- `POST /api/score-pronunciation-simple` - 简化评分
- `POST /api/score-pronunciation-detailed` - 详细分析

#### 语法检测
- `POST /api/check-grammar-text` - 文本语法检测
- `POST /api/transcribe-audio` - 语音转文字

#### 练习管理
- `GET /api/exercise-sets` - 获取练习集列表
- `POST /api/exercise-sets` - 创建练习集
- `POST /api/custom-exercise` - 获取练习题目
- `POST /api/exercise-results` - 记录练习结果

### API请求示例

**发音评分请求**：
```javascript
const formData = new FormData();
formData.append('reference_text', 'Hello world');
formData.append('audio_file', audioBlob);

fetch('/api/score-pronunciation', {
    method: 'POST',
    body: formData
}).then(response => response.json());
```

**语法检测请求**：
```javascript
fetch('/api/check-grammar-text', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        text: 'I have went to the store yesterday.'
    })
}).then(response => response.json());
```

## 🗄️ 数据库结构

### 主要数据表

**users（用户表）**
- 存储用户基本信息、登录凭据
- 包含学习偏好和语言设置

**learning_records（学习记录表）**
- 记录用户的学习活动
- 包含练习类型、得分、详细结果

**exercise_sets（练习集表）**
- 存储自定义练习集信息
- 支持公开和私有练习集

**exercise_items（练习项目表）**
- 存储具体的练习题目
- 包含内容、难度、标签等信息

**user_progress（用户进度表）**
- 跟踪用户学习进度
- 记录完成状态和最佳成绩

## 🔒 安全特性

- JWT令牌认证
- 密码加密存储
- SQL注入防护
- XSS攻击防护
- 文件上传安全检查

## 🎛️ 部署指南

### 生产环境部署

1. **环境准备**
```bash
# 安装系统依赖
sudo apt update
sudo apt install python3 python3-pip mysql-server ffmpeg

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate
```

2. **应用配置**
```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
```

3. **数据库初始化**
```sql
CREATE DATABASE english_assistant;
CREATE USER 'app_user'@'localhost' IDENTIFIED BY 'secure_password';
GRANT ALL PRIVILEGES ON english_assistant.* TO 'app_user'@'localhost';
```

4. **启动服务**
```bash
# 使用Gunicorn启动
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🚨 故障排除

### 常见问题

**1. 模型加载失败**
- 检查模型文件是否完整下载
- 确认模型路径配置正确
- 验证系统内存是否充足

**2. 音频录制问题**
- 检查浏览器麦克风权限
- 确认音频设备正常工作
- 验证HTTPS连接（某些浏览器要求）

**3. 数据库连接错误**
- 检查数据库服务是否启动
- 验证连接参数配置
- 确认数据库用户权限

**4. LanguageTool API错误**
- 检查网络连接
- 验证API密钥（如使用付费版）
- 考虑使用本地部署

### 性能优化

**1. 数据库优化**
- 添加适当的索引
- 定期清理历史数据
- 使用连接池

**2. 模型优化**
- 使用GPU加速（如有条件）
- 模型量化减少内存占用
- 批处理请求

**3. 缓存策略**
- Redis缓存用户会话
- 静态资源CDN加速
- API响应缓存

## 📈 监控和日志

### 日志配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### 监控指标

- 请求响应时间
- 错误率统计
- 用户活跃度
- 模型推理性能
- 数据库查询性能

## 🤝 贡献指南

### 开发环境设置

1. Fork项目仓库
2. 创建功能分支
3. 提交代码变更
4. 创建Pull Request

### 代码规范

- 遵循PEP 8 Python编码规范
- 添加适当的注释和文档
- 编写单元测试
- 确保代码通过所有测试

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- 项目维护者：鸭子先生团队
- 问题反馈：通过GitHub Issues
- 邮箱：[1977358865@qq.com]

## 🙏 致谢

感谢以下开源项目的支持：
- [Flask](https://flask.palletsprojects.com/)
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [LibROSA](https://librosa.org/)
- [LanguageTool](https://languagetool.org/)
- [Whisper](https://openai.com/research/whisper)

---

## 📝 更新日志

### v2.0.1 (2025-10-12)
- 修复HTML结构错误
- 优化音频处理性能
- 增强错误处理机制
- 完善API文档

### v2.0.0 (2025-9-20)
- 添加音素级发音评分
- 重构数据库设计
- 实现用户认证系统
- 优化前端界面

### v1.0.0 (2024-12-01)
- 初始版本发布
- 基础发音评分功能
- 语法检测功能
- 简单Web界面

---

*最后更新：2025年10月12日*