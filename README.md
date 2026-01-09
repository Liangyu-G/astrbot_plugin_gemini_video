# AstrBot Gemini Video Plugin 🎬

[![AstrBot](https://img.shields.io/badge/AstrBot-v3.0+-blue.svg)](https://github.com/Soulter/AstrBot)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 💬 **加入QQ群获取支持**：[点击加入 308749810](https://qun.qq.com/universal-share/share?ac=1&authKey=fqH3xeXAFOlK6983ACfvuMn1uK9cnK8KEwU9p7mprX3DXHncn3uVx%2BcnGmphv%2BZK&busi_data=eyJncm91cENvZGUiOiIzMDg3NDk4MTAiLCJ0b2tlbiI6Ilp4dnk1anEySG9Wdzg1NEI0ZmZ2NkVYcXIwb3QvQ3VNKzNwMGNLYlQ4aGlvTzdJdUJDY011V3hXUjI0cUlRMTUiLCJ1aW4iOiIyMTQ4MTkzNDU4In0=&data=6bw6cjWn50SQWIAWQ9nhxMn1xCHo62sPn4oJP3qWfes7lRZXBAZq1UPnfa31fKPMhMZcQxuIqDA4IildMyb6u_TQHVCntNf4tvSZeFN6D9U&svctype=5&tempid=h5_group_info) 🚀

一个为 AstrBot 设计的**高性能、高可靠性** Gemini 视频分析插件。支持多种上传模式、自动压缩、智能网络优化，完美兼容 Agent 工具调用与人格注入。

> 🎯 **优化重点**：经过深度优化的下载逻辑、上传可靠性、视频压缩，以及针对 QQ/腾讯 CDN 的智能直连优化。

## ✨ 核心特性

### 🚀 智能上传策略
- **Base64 编码上传**：适用于中小型视频（≤30MB），兼容性最广，能有效绕过部分中转服务商的文件上传限制
- **File API 上传**：适用于大型视频，先上传到 API 服务器获取 CDN URL，再进行分析
- **自动压缩**：视频超过阈值（默认25MB）时自动调用 `ffmpeg` 压缩为 720p/CRF28，大幅提高上传成功率

### 🎭 深度人格注入
- 采用 **Double Injection** 策略，将 Gemini 的客观分析与用户配置的人格设定深度融合
- 确保机器人的回复既专业准确，又符合其角色性格（如亚托莉的少女萝卜子语气）

### 🔄 强制刷新机制
- 每次分析都触发**强制重新下载**和**重新分析**
- 不依赖过时的本地缓存，确保获取视频的最新内容
- 适用于实时更新的视频内容场景

### 🌐 网络优化与可靠性
- **智能代理绕过**：自动检测 QQ/腾讯 CDN 域名（`.qq.com`、`.qq.com.cn`、`.tencent.com`），强制直连以获得最佳速度
- **多轮重试机制**：下载、上传均支持可配置的重试次数和间隔
- **超时监控**：实时监控下载速度和上传进度，防止卡死
- **OneBot 协议兼容**：支持 `get_file` 和 `download_file` 接口，解析文件 ID 为真实 URL/路径

### 🤖 Agent 工具集成
- **函数工具调用**：注册为 `gemini_analyze_video` 工具，LLM 可自动调用
- **Reply 消息优先**：当用户回复（Reply）包含视频的消息时，优先分析被引用的视频，而非从聊天历史中错误提取
- **即时反馈**：调用工具时立即发送"正在分析..."提示（可自定义 `watching_hint`）

### 🧹 自动维护
- 后台清理任务定期删除过期视频（默认3天）
- 支持自定义清理间隔（默认6小时）
- 磁盘空间无忧

### 💬 完美配套
- 兼容 `Splitter` 插件（分段回复）
- 兼容 AstrBot 内置 TTS 系统
- 支持流式响应和 SSE 格式

---

## 📦 安装方法

### 方式一：通过 AstrBot 管理面板（推荐）
1. 打开 AstrBot Web 管理面板
2. 导航到"插件" → "安装插件"
3. 输入仓库地址：
   ```text
   https://github.com/Liangyu-G/astrbot_plugin_gemini-video
   ```
4. 等待安装完成，重启机器人

### 方式二：手动安装
```bash
cd AstrBot/data/plugins
git clone https://github.com/Liangyu-G/astrbot_plugin_gemini-video
# 重启 AstrBot
```

### 依赖要求
- **FFmpeg**（可选，但强烈推荐）：用于视频压缩功能
  - Windows: 下载 FFmpeg，添加到 PATH
  - Linux/macOS: `apt install ffmpeg` 或 `brew install ffmpeg`

---

## ⚙️ 配置详解

### 基本配置

| 配置项 | 说明 | 默认值 | 备注 |
|:---|:---|:---|:---|
| `base_url` | Gemini API 基础 URL | *必填* | 支持官方 API 和第三方中转（如柏拉图） |
| `api_key` | API 密钥 | *必填* | 确保有充足的配额 |
| `model` | 使用的模型 | `gemini-3-flash-preview` | 推荐 flash 系列，速度快 |
| `max_tokens` | 最大生成 Token 数 | `4000` | 分析结果的长度限制 |

### 上传模式配置

| 配置项 | 说明 | 默认值 | 备注 |
|:---|:---|:---|:---|
| `upload_mode` | 上传模式 | `base64` | `base64` 或 `file_api` |
| `max_base64_size_mb` | Base64 模式最大文件大小 | `30` | 建议 20-30MB |
| `upload_retries` | 上传重试次数 | `3` | 网络不稳定时可增加 |
| `upload_stream_timeout` | 上传流超时（秒） | `3600` | File API 模式推荐 ≥600s |

### 视频压缩配置

| 配置项 | 说明 | 默认值 | 备注 |
|:---|:---|:---|:---|
| `enable_compression` | 是否启用自动压缩 | `true` | 需要安装 FFmpeg |
| `compression_threshold_mb` | 压缩触发阈值（MB） | `25` | 超过此大小自动压缩 |

**压缩策略**：
- 分辨率：缩放到 720p（保持宽高比）
- 编码：H.264，CRF 28（平衡质量与文件大小）
- 通常可减小文件 60-80%

### 下载与网络配置

| 配置项 | 说明 | 默认值 | 备注 |
|:---|:---|:---|:---|
| `download_retries` | 下载重试次数 | `3` | - |
| `download_retry_delay` | 重试间隔（秒） | `5` | - |
| `download_stream_timeout` | 下载流超时（秒） | `300` | 大视频可适当增加 |
| `proxy` | HTTP 代理地址 | *空* | 格式：`http://host:port` |

**智能代理绕过**：
- QQ/腾讯 CDN 域名（如 `multimedia.nt.qq.com.cn`）会自动绕过代理，强制直连
- 提高国内视频下载速度

### 其他配置

| 配置项 | 说明 | 默认值 |
|:---|:---|:---|
| `default_prompt` | 默认分析提示词 | *详细分析提示* |
| `watching_hint` | 分析中的提示文本 | `亚托莉正在看视频哦~` |
| `video_retention_days` | 视频保留天数 | `3` |
| `cleanup_interval_hours` | 清理检查间隔（小时） | `6` |
| `max_video_size_mb` | 允许的最大视频大小 | `100` |

---

## 📖 使用指南

### 方式一：命令触发

**直接分析视频**：
```
1. 发送视频到群/私聊
2. 发送命令：/分析视频
```

**引用视频分析**：
```
1. 回复（Reply）包含视频的消息
2. 发送：/分析视频 请分析这个视频的剪辑技巧
```

**自定义提示词**：
```
/分析视频 分析这个视频的音乐风格和情绪表达
```

### 方式二：LLM 工具调用（推荐）

如果您的主 LLM 支持 Function Calling（如 GPT-4、Claude、Gemini），可以直接在对话中说：

```
- "帮我看看这个视频讲了什么"
- "分析一下刚才那个视频的拍摄手法"
- "这个视频里的 AI 在讨论什么话题？"
```

**工作流程**：
1. LLM 识别用户意图，调用 `gemini_analyze_video` 工具
2. 插件发送"正在分析..."提示（立即反馈）
3. 下载视频 → 压缩（如需要）→ 上传 → 调用 Gemini API
4. 返回分析结果给 LLM
5. LLM 基于分析结果，以设定的人格语气回复用户

**Reply 消息优先机制**：
- 当您回复（Reply）一个包含视频的消息时，插件**优先分析被引用的视频**
- 即使 LLM 从聊天历史中提取了其他 URL，也会被忽略
- 确保"分析的就是您想要的那个视频"

---

## 🔧 工作原理

### 视频定位逻辑
```
1. 优先检查 Reply 消息中的视频（从缓存或 chain 获取）
2. 检查当前消息中的视频
3. LLM 提供的 URL（仅作为后备）
```

### 下载流程
```
1. 检测 URL 类型：
   - file:/// → 直接使用本地路径
   - HTTP URL → 进入下载流程
   - 文件名/ID → 调用 OneBot get_file 解析
   
2. 智能代理判断：
   - QQ/腾讯域名 → 强制直连
   - 其他域名 → 使用配置的 proxy
   
3. 下载监控：
   - 每10秒检查速度
   - 停滞检测（30秒无数据传输则报错）
   - 支持多轮重试
```

### 上传与分析流程
```
1. 检查文件大小
2. 触发压缩（如超过阈值）
3. 根据 upload_mode 选择模式：
   
   Base64 模式：
   - 检查大小 ≤ max_base64_size_mb
   - Base64 编码
   - 构造 data URI
   - 调用 OpenAI 兼容 API
   
   File API 模式：
   - 上传到 /v1/files 端点
   - 获取 file_uri
   - 轮询文件状态（等待 ACTIVE）
   - 调用 Gemini Native API
```

### 人格注入策略（Double Injection）
```python
final_prompt = f"""
[System Instruction: You are {system_prompt}]
[Context: The user sent a video. Here is a description of the video content:]

{gemini_analysis_result}

[User Request: {user_prompt}]

[Task: Reply to the User Request based on the video description. 
Important: You must ACT AS your persona. Do NOT act as an AI assistant. Stay in character.]
"""
```

---

## ❓ 常见问题与故障排查

### Q1: 提示"视频分析失败，未能获取分析结果"

**可能原因**：
1. API Key 无效或配额不足
2. Base URL 配置错误
3. 网络连接问题（无法访问 API）

**解决方案**：
```
1. 检查 api_key 是否正确
2. 尝试访问 base_url（如 curl https://api.bltcy.ai）
3. 如使用代理，检查 proxy 配置
4. 查看日志中的详细错误信息
```

### Q2: 下载速度极慢（<50 KB/s）

**可能原因**：
1. QQ CDN 域名未被正确识别为国内域名
2. 网络环境限制（运营商 QoS）
3. 服务器端限流

**解决方案**：
```
1. 查看日志，确认是否有"检测到国内域名，强制直连"
2. 如果没有，检查 main.py:559 的域名匹配逻辑
3. 尝试清空 proxy 配置（国内视频不需要代理）
4. 考虑使用 OneBot 的 download_file API（如支持）
```

### Q3: 上传失败，提示 77% 处停滞

**可能原因**：
- 服务器端超时（通常为 120 秒）
- 本地代理的超时限制

**解决方案**：
```
1. 启用视频压缩：enable_compression = true
2. 减小 compression_threshold_mb（如 20）
3. 切换到 file_api 模式（如中转支持）
4. 增加 upload_stream_timeout
5. 如使用本地代理，调整代理的超时设置
```

### Q4: FFmpeg 未找到，压缩失败

**现象**：
```
[Gemini Video] FFmpeg not found, skipping compression
```

**解决方案**：
```bash
# Windows
下载 FFmpeg，解压后将 bin 目录添加到系统 PATH

# Linux
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# 验证安装
ffmpeg -version
```

### Q5: Base64 模式提示文件过大

**错误示例**：
```
❌ 视频文件过大 (35.2MB)，Base64 模式最大支持 30MB
```

**解决方案**：
```
方案一：启用压缩
- enable_compression: true
- compression_threshold_mb: 25

方案二：切换上传模式
- upload_mode: "file_api"
（需要 API 支持 /v1/files 端点）

方案三：调整限制（不推荐）
- max_base64_size_mb: 50
（可能导致请求失败）
```

### Q6: 分析的是错误的视频（聊天历史中的旧视频）

**原因**：
- LLM 从聊天历史中提取了旧视频的 URL

**解决方案**：
```
插件已修复此问题（v1.1.0+）：
- 现在优先检查 Reply 消息中的视频
- LLM 提供的 URL 仅作为后备
- 确保更新到最新版本
```

### Q7: OneBot 平台视频下载失败

**错误示例**：
```
UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol
```

**原因**：
- OneBot 返回的是文件 ID（如 `abc123.mp4`），不是 URL

**解决方案**：
```
插件已自动处理（v1.1.0+）：
- 自动调用 get_file 接口解析文件 ID
- 确保 OneBot 实现（如 LLOneBot）支持 get_file
- 查看日志确认是否成功解析
```

---

## 🔍 配置示例

### 示例一：国内环境 + 柏拉图 API（推荐）
```json
{
  "base_url": "https://api.bltcy.ai",
  "api_key": "your_api_key_here",
  "model": "gemini-3-flash-preview",
  "upload_mode": "base64",
  "max_base64_size_mb": 30,
  "enable_compression": true,
  "compression_threshold_mb": 20,
  "proxy": "",
  "upload_retries": 3,
  "download_retries": 3
}
```

### 示例二：国外环境 + 官方 API
```json
{
  "base_url": "https://generativelanguage.googleapis.com",
  "api_key": "AIza...",
  "model": "gemini-2.5-flash",
  "upload_mode": "file_api",
  "enable_compression": true,
  "proxy": "http://127.0.0.1:7890",
  "upload_stream_timeout": 600
}
```

### 示例三：网络不稳定环境
```json
{
  "upload_retries": 5,
  "download_retries": 5,
  "download_retry_delay": 10,
  "enable_compression": true,
  "compression_threshold_mb": 15,
  "max_base64_size_mb": 20
}
```

---

## 🛠️ 技术细节

### 依赖库
- `httpx`: HTTP 客户端（支持流式下载/上传、代理）
- `asyncio`: 异步 I/O
- `ffmpeg`: 视频压缩（外部依赖）
- `pathlib`: 路径操作
- `base64`: Base64 编码

### 性能特性
- **异步下载**：使用 `aiter_bytes()` 流式下载，内存友好
- **并发控制**：URL 级别的下载锁，防止重复下载
- **智能缓存**：消息级别的视频路径缓存
- **后台清理**：独立的 `asyncio.Task` 定期清理过期文件

### 安全特性
- API Key 日志脱敏（显示为 `********`）
- 临时文件自动清理
- 文件大小限制（防止磁盘占用过大）

---

## 📝 更新日志

### v1.1.0 (2026-01-10)
- ✨ 新增：自动视频压缩（ffmpeg）
- ✨ 新增：可配置的上传重试和 Base64 大小限制
- 🐛 修复：Reply 消息优先级问题（不再分析错误的视频）
- 🐛 修复：QQ CDN 域名检测（支持 `.qq.com.cn`）
- 🐛 修复：OneBot 文件 ID 解析（支持 `get_file`）
- 🐛 修复：Agent 工具调用的即时反馈
- ⚡ 优化：智能代理绕过逻辑
- ⚡ 优化：下载速度监控和停滞检测
- 📝 文档：全面更新 README 和配置说明

### v1.0.0 (初始版本)
- 基础功能实现

---

## 🤝 贡献与支持

### 💬 QQ交流群
**遇到问题？想要交流经验？加入我们的QQ群！**

- **群号**：308749810
- **加群链接**：[点击加入](https://qun.qq.com/universal-share/share?ac=1&authKey=fqH3xeXAFOlK6983ACfvuMn1uK9cnK8KEwU9p7mprX3DXHncn3uVx%2BcnGmphv%2BZK&busi_data=eyJncm91cENvZGUiOiIzMDg3NDk4MTAiLCJ0b2tlbiI6Ilp4dnk1anEySG9Wdzg1NEI0ZmZ2NkVYcXIwb3QvQ3VNKzNwMGNLYlQ4aGlvTzdJdUJDY011V3hXUjI0cUlRMTUiLCJ1aW4iOiIyMTQ4MTkzNDU4In0=&data=6bw6cjWn50SQWIAWQ9nhxMn1xCHo62sPn4oJP3qWfes7lRZXBAZq1UPnfa31fKPMhMZcQxuIqDA4IildMyb6u_TQHVCntNf4tvSZeFN6D9U&svctype=5&tempid=h5_group_info) 🚀

在群里可以：
- 💡 快速获取技术支持
- 🐛 反馈Bug和问题
- 💬 与其他用户交流使用经验
- 🎯 提出功能建议
- 📢 获取最新版本发布通知

### 问题反馈
遇到问题？请提交 Issue：
https://github.com/Liangyu-G/astrbot_plugin_gemini-video/issues

### 功能建议
欢迎提交 Feature Request 或 Pull Request！

### 测试环境
- ✅ LLOneBot + 柏拉图 API
- ⚠️ 其他平台未充分测试，欢迎反馈兼容性问题

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**Love and Robots — 毕竟，亚托莉可是高性能的嘛！** 🤖💖
