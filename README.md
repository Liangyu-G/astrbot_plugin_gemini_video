# AstrBot Gemini Video Plugin 🎬

[![AstrBot](https://img.shields.io/badge/AstrBot-v3.0+-blue.svg)](https://github.com/Soulter/AstrBot)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个为 AstrBot 设计的高性能 Gemini 视频分析插件。支持多种上传方式、自动清理、人格注入，并完美兼容分段回复与 TTS 功能。
有个问题是代码写死了解读提示，请自行更改。
仅在LLBOT平台测试过，其他机器人未测试，不保证其他平台的兼容性。

## ✨ 功能亮点

- **🚀 Base64 编码上传**：使用 Base64 编码方式直接发送视频数据，兼容性最广，能有效绕过部分中转服务商的限制。支持 30MB 以下的视频文件。
- **🎭 深度人格注入**：采用 **Double Injection** 策略，将 Gemini 的分析结果与用户人格配置深度结合，让机器人的回复不仅专业，而且更符合其设定的性格。
- **🔄 强制刷新逻辑**：每次分析请求都会触发强制重新下载和重新分析，确保获取视频的最新内容，不依赖过时的本地缓存。
- **🧹 自动维护**：内置后台清理任务，根据配置定期删除过期的视频文件，磁盘空间无忧。
- **🌐 强大的网络兼容性**：支持自定义代理、多轮下载重试（可配置间隔）以及 OneBot 协议的 `download_file` 与 `get_file` 事件。
- **💬 实时反馈**：在分析过程中提供"正在看视频"的中间反馈（可自定义），提升用户交互体验。
- **🧩 完美配套**：兼容 `Splitter`（分段回复）插件和 AstrBot 内置 TTS 系统。

## 📦 安装方法

在 AstrBot 管理面板中，点击“插件” -> “安装插件”，输入以下地址：

```text
https://github.com/Liangyu-G/astrbot_plugin_gemini-video
```

安装完成后重启机器人。

## ⚙️ 配置项说明

| 配置项 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `base_url` | Gemini API 的基础 URL。支持官方和第三方中转地址。 | (必填) |
| `api_key` | 填入您的 Gemini API Key。 | (必填) |
| `model` | 建议选择 `gemini-3-flash-preview` 或更高版本。 | `gemini-3-flash-preview` |
| `max_video_size_mb` | 允许分析的最大视频大小。超过将报错。 | `100` |
| `download_retries` | 下载视频失败后的最大重试次数。 | `3` |
| `download_retry_delay`| 每次重试下载之前的等待间隔（秒）。 | `5` |
| `video_retention_days`| 视频文件在本地保留的天数。设置 0 禁用清理。 | `3` |
| `proxy` | 用于连接 API 或下载视频的 HTTP 代理地址。 | (空) |

## 📖 使用指南

### 1. 命令触法
直接发送视频文件到群内或私聊，然后发送 `/分析视频` 或回复该视频发送 `/分析视频 [可选提示词]`。

### 2. 工具调用 (LLM 驱动)
如果您的模型支持 Tool Call（函数调用），您可以直接在对话中说：
- “帮我看看这个视频里讲了什么”
- “评价一下刚才那个视频里的操作”

机器人会自动通过 `gemini_analyze_video` 工具提取视频概要并以设定的语气回复您。

## ❓ 常见问题与排查 (FAQ)

### Q1: 机器人提示“Base64 和 Native 流程均不可用”
- **原因**：通常是网络连通性问题或 API Key 额度不足。
- **解决**：检查 `base_url` 是否正确，确认 `proxy` 是否能正常访问 Google 服务。如果是中转接口，请确认其是否支持 `v1beta/files` 上传接口。

### Q2: 下载失败，显示 ConnectError
- **原因**：视频链接所在服务器由于某些原因无法直连。
- **解决**：在配置中正确填写 `proxy`。如果使用的是 OneBot 协议的服务端（如 Go-CQHTTP / LLOneBot），请确保其网络环境良好。

### Q3: 为什么大视频分析很慢？
- **原因**：为了保证稳定性，大视频会触发 Native 上传模式，包含：下载 -> 上传 -> 状态轮询 -> 分析模型响应。
- **解决**：请在配置中调大 `timeout`（建议 300s 以上）。

---
*Love and Robots — 毕竟，亚托莉可是高性能的嘛！*
