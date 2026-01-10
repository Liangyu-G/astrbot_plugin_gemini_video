from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger
from astrbot.api.message_components import *
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent


from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.agent.run_context import ContextWrapper
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
import httpx
import httpcore
import os
import json
import asyncio
import shutil
import time
import uuid
import aiofiles
import base64
import mimetypes
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Any


@pydantic_dataclass
class GeminiVideoAnalysisTool(FunctionTool[AstrAgentContext]):
    """Gemini Video Analysis Tool"""
    name: str = "gemini_analyze_video"
    description: str = "Use Gemini vision model to analyze a video and return a description. Use this when the user sends a video or provides a video URL and asks about it."
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "video_url": {
                    "type": "string",
                    "description": "URL of the video to analyze.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Specific question about the video content.",
                },
            },
        }
    )
    
    plugin: Any = None # Should be GeminiVideoPlugin instance

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        if not self.plugin:
            return "Plugin instance missing."
        
        
        video_url = kwargs.get("video_url")
        prompt = kwargs.get("prompt", "Describe this video.")
        
        # IMPORTANT: Prioritize finding video from the current event/Reply
        # The LLM may hallucinate or extract URLs from chat history
        video_comp = await self.plugin._find_video_component(context.context.event)
        if video_comp:
            video_url = video_comp.file
            logger.info(f"[Gemini Video] Found video in current event: {video_url}")
        elif not video_url:
            # Only error out if there's truly no video anywhere
            return "Please provide a video URL or send a video first."
        else:
            # LLM provided a URL, but no video in current event
            # This might be a hallucination or extracted from history
            logger.warning(f"[Gemini Video] Using LLM-provided URL (no video in current event): {video_url}")

        try:

             logger.info(f"[Gemini Video] Tool call started. URL: {video_url}, Prompt: {prompt}")
             
             # Notify user that analysis is starting
             try:
                 hint = self.plugin.config.get("watching_hint", "⏳ 正在分析视频内容，请稍候...")
                 yield_msg = MessageChain([Plain(hint)])
                 await self.plugin.context.send_message(context.context.event.unified_msg_origin, yield_msg)
             except Exception as e:
                 logger.warning(f"[Gemini Video] Failed to send analyzing status: {e}")
             
             # Pass the event context to the analysis method for more robust downloading
             logger.info(f"[Gemini Video] Calling _perform_video_analysis...")
             
             result = await self.plugin._perform_video_analysis(video_url, prompt, event=context.context.event)
             logger.info(f"[Gemini Video] _perform_video_analysis returned. Length: {len(result) if result else 0}") 
             
             if not result:
                 return "视频分析失败，未能获取分析结果。"
                 
             return result
        except Exception as e:
            return f"Error analyzing video: {str(e)}"

@register("astrbot_plugin_gemini_video", "liangyu", "Gemini 视频分析插件", "1.0.0")
class GeminiVideoPlugin(Star):
    """Gemini 视频分析插件"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or AstrBotConfig()
        self.client: Optional[httpx.AsyncClient] = None
        self.video_storage_path: Optional[Path] = None
        self.video_cache: dict[str, str] = {} # message_id -> local_path
        
        # Register tool
        self.context.add_llm_tools(GeminiVideoAnalysisTool(plugin=self))

    async def initialize(self):
        """初始化插件"""
        # 加载配置
        logger.info(f"[Gemini Video] 配置加载完成: {self.config}")
        
        # 视频缓存: Map[LocalPath, AnalysisResult]
        self.video_analysis_cache: dict[str, str] = {}
        
        # 并发控制：正在下载的 URL 集合
        self._downloading_urls = set()
        
        # 视频路径缓存: Map[MessageID, LocalPath]
        self.video_path_cache: dict[str, str] = {}
        
        # 并发控制：使用信号量替代互斥锁，允许一定程度的并发
        max_concurrent = self.config.get("max_concurrent_analysis", 3)
        self.concurrency_limiter = asyncio.Semaphore(max_concurrent)
        logger.info(f"[Gemini Video] 并发限制设置为: {max_concurrent}")

        # 创建视频存储目录（使用官方推荐的数据目录，而非插件代码目录）
        storage_path = self.config.get("video_storage_path", "videos")
        if storage_path:
            # 使用 StarTools 获取官方数据目录，遵循代码与数据分离的最佳实践
            data_dir = StarTools.get_data_dir(self.name)
            self.video_storage_path = data_dir / storage_path
            self.video_storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Gemini Video] 视频存储路径: {self.video_storage_path}")

        # 初始化 HTTP 客户端
        proxy = self.config.get("proxy", "")
        timeout = httpx.Timeout(
            self.config.get("timeout", 300),
            connect=30.0,
        )
        
        # 根据是否有代理配置创建客户端
        if proxy:
            self.client = httpx.AsyncClient(
                timeout=timeout,
                proxy=proxy,
                follow_redirects=True,
            )
        else:
            self.client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
            )
        # 启动清理任务并保存引用，防止被垃圾回收
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("[Gemini Video] 插件初始化完成")

    async def _cleanup_loop(self):
        """后台清理任务循环"""
        while True:
            try:
                await self._do_cleanup()
            except Exception as e:
                logger.error(f"[Gemini Video] Cleanup task error: {e}")
            
            interval = self.config.get("cleanup_interval_hours", 6)
            await asyncio.sleep(max(1, interval) * 3600)

    async def _do_cleanup(self):
        """执行清理逻辑"""
        retention_days = self.config.get("video_retention_days", 3)
        if retention_days <= 0:
            return

        if not self.video_storage_path or not self.video_storage_path.exists():
            return

        now = time.time()
        expiry_seconds = retention_days * 86400
        
        count = 0
        for item in self.video_storage_path.iterdir():
            if item.is_file() and item.suffix.lower() in (".mp4", ".ts", ".mkv"):
                mtime = item.stat().st_mtime
                if now - mtime > expiry_seconds:
                    try:
                        item.unlink()
                        count += 1
                        # 同时也从内存缓存中移除 (如果存在)
                        keys_to_del = [k for k, v in self.video_path_cache.items() if v == str(item)]
                        for k in keys_to_del: self.video_path_cache.pop(k, None)
                    except Exception as e:
                        logger.warning(f"[Gemini Video] 无法删除过期文件 {item}: {e}")
        
        if count > 0:
            logger.info(f"[Gemini Video] 自动清理完成，已删除 {count} 个过期视频文件。")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message_receive(self, event: AstrMessageEvent):
        """监听消息，自动缓存视频，并根据需要预解析"""
        if not event.message_obj.message:
            return

        video_comp = None
        for comp in event.message_obj.message:
            if isinstance(comp, Video):
                video_comp = comp
                break
        
        if video_comp:
            try:
                # 1. 仅缓存视频 URL，不自动下载
                # 获取 URL: 优先使用属性，其次使用 file 字段
                url = getattr(video_comp, "url", None) or video_comp.file
                if url:
                    msg_id = str(event.message_obj.message_id)
                    self.video_path_cache[msg_id] = url
                    logger.debug(f"[Gemini Video] 已缓存视频消息 URL {msg_id} -> {url}")
                
            except Exception as e:
                logger.warning(f"[Gemini Video] 自动缓存视频 URL 失败: {e}")

    async def _find_video_component(self, event: AstrMessageEvent) -> Video | None:
        """从消息或引用中查找视频组件"""
        if not event or not event.message_obj or not event.message_obj.message:
            return None
            
        # 1. 检查引用消息
        for comp in event.message_obj.message:
            if isinstance(comp, Reply):
                # 尝试从缓存获取
                if str(comp.id) in self.video_path_cache:
                    cached_val = self.video_path_cache[str(comp.id)]
                    if cached_val.startswith("http"):
                        # 是 URL，返回包含 URL 的 Video 组件
                        logger.debug(f"[Gemini Video] 从缓存恢复视频 URL: {cached_val}")
                        return Video(file=cached_val)
                    elif os.path.exists(cached_val):
                        # 是本地路径
                        return Video(file=cached_val, path=cached_val)
                
                # 尝试从引用的 chain 中找
                if comp.chain:
                    for sub_comp in comp.chain:
                        if isinstance(sub_comp, Video):
                            return sub_comp
        
        # 2. 检查当前消息
        for comp in event.message_obj.message:
            if isinstance(comp, Video):
                return comp
                
        return None

    @filter.command("分析视频")
    async def analyze_video(self, event: AstrMessageEvent, prompt: str = ""):
        """分析视频指令处理"""
        if not prompt:
            prompt = self.config.get(
                "default_prompt", "请详细分析这个视频的内容，包括场景、人物、动作和主题。"
            )

        try:
            # 获取视频组件
            video_component = await self._find_video_component(event)
            
            if not video_component:
                yield event.plain_result(
                    "❌ 请发送视频文件或引用包含视频的消息后再使用此命令。\n\n"
                    "使用方法：\n"
                    "1. 先发送视频，然后发送 /分析视频\n"
                    "2. 或者引用包含视频的消息，然后发送 /分析视频"
                )
                return
            
            watching_hint = self.config.get("watching_hint", "亚托莉正在看视频哦~")
            yield event.plain_result(watching_hint)

            video_url = video_component.file
            if not video_url:
                yield event.plain_result("❌ 无法获取视频路径。")
                return

            # 获取分析结果
            gemini_analysis_result = await self._perform_video_analysis(video_url, prompt, event=event)
            
            # 检查是否包含错误信息
            if gemini_analysis_result.startswith("❌") or ("失败" in gemini_analysis_result and len(gemini_analysis_result) < 100):
                # 优雅地告知用户分析失败，而不是让 LLM 瞎编
                error_msg = gemini_analysis_result.replace("❌", "").strip()
                yield event.plain_result(f"💡 视频分析遇到了一点小问题：\n{error_msg}\n\n请稍后再试一次吧！")
                return

            # 调用主模型进行生成
            try:
                # 获取当前会话使用的 LLM Provider ID
                provider_id = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
                
                # 获取当前会话的人格设置
                personality = await self.context.persona_manager.get_default_persona_v3(event.unified_msg_origin)
                system_prompt = personality['prompt']
                # begin_dialogs 用于设定语气（公开 API）
                contexts = personality['begin_dialogs']

                # 处理空 Prompt
                final_user_prompt = prompt if prompt.strip() else "Look at this video."

                # 构建给主模型的 Prompt
                # 注意：system_prompt 会作为单独参数传入，不需要在这儿重复
                final_prompt = (
                    f"[Context: The user sent a video. Here is a description of the video content:]\n\n"
                    f"{gemini_analysis_result}\n\n"
                    f"[User Request: {final_user_prompt}]\n\n"
                    f"[Task: Reply to the User Request based on the video description. Stay in character as defined in your system prompt.]"
                )
                
                # 调用主模型
                llm_response = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=final_prompt,
                    system_prompt=system_prompt,
                    contexts=contexts
                )
                
                setattr(event, "__is_llm_reply", True) # 标记为 LLM 回复，以便 Splitter 插件处理
                
                result = event.plain_result(llm_response.completion_text)
                result.set_result_content_type(ResultContentType.LLM_RESULT)
                yield result
                
            except Exception as e_llm:
                logger.error(f"[Gemini Video] 调用主模型失败: {e_llm}", exc_info=True)
                # 降级：直接返回 Gemini 的结果
                yield event.plain_result(f"⚠️ 主模型调用失败，显示原始分析结果：\n\n{gemini_analysis_result}")

        except Exception as e:
            logger.error(f"[Gemini Video] 处理出错: {e}", exc_info=True)
            yield event.plain_result(f"❌ 处理失败: {str(e)}")

    async def _perform_video_analysis(self, video_url: str, prompt: str | None = None, event: AstrMessageEvent = None) -> str:
        """执行视频分析的核心逻辑：先下载，再根据模式选择上传方式"""
        logger.info(f"[Gemini Video] _perform_video_analysis entered with URL: {video_url}")
        try:
            # 第一步：下载视频到本地（使用新的 _download_video 方法）
            local_path = ""
            is_temp = False
            
            # 判断是否是默认提示词
            default_prompt = self.config.get("default_prompt", "请详细分析这个视频的内容，包括场景、人物、动作和主题。")
            is_default_prompt = (prompt is None) or (prompt.strip() == default_prompt.strip()) or (prompt.strip() == "Describe this video.")
            
            if video_url.startswith("file:///"):
                local_path = video_url[8:]
                logger.info(f"[Gemini Video] Using local file path: {local_path}")
            elif os.path.exists(video_url) and os.path.isfile(video_url):
                local_path = video_url
                logger.info(f"[Gemini Video] Using existing local file: {local_path}")
            else:
                # 使用新的 _download_video 方法处理所有下载逻辑
                try:
                    dummy_video = Video(file=video_url)
                    stored_path = await self._download_video(dummy_video, event)
                    if stored_path and os.path.exists(stored_path):
                        local_path = stored_path
                        is_temp = False  # _download_video 已经存储到永久目录
                        logger.info(f"[Gemini Video] Download successful: {local_path}")
                except Exception as e_dl:
                    logger.error(f"[Gemini Video] Download failed: {e_dl}", exc_info=True)
                    return f"❌ 无法下载视频: {str(e_dl)}"
            
            if not local_path or not os.path.exists(local_path):
                return "❌ 视频文件不存在或下载失败。"

            # 检查文件大小
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            max_size = self.config.get("max_video_size_mb", 100)
            if file_size_mb > max_size:
                return f"❌ 视频文件过大 ({file_size_mb:.1f}MB)，最大支持 {max_size}MB。"

            logger.info(f"[Gemini Video] Video ready at {local_path}, size: {file_size_mb:.1f}MB")
            
            # 尝试自动压缩
            original_size_mb = file_size_mb  # 保存原始大小
            try:
                compressed_path = await self._compress_video_if_needed(local_path)
                if compressed_path != local_path:
                    compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                    
                    # 只有压缩后文件更小才使用
                    if compressed_size_mb < original_size_mb:
                        logger.info(f"[Gemini Video] ✅ 压缩成功，使用压缩后的视频: {compressed_path} ({original_size_mb:.1f}MB → {compressed_size_mb:.1f}MB)")
                        local_path = compressed_path
                        is_temp = True # 标记为临时文件，确保会被清理
                        file_size_mb = compressed_size_mb
                    else:
                        logger.warning(f"[Gemini Video] ⚠️ 压缩后反而变大 ({original_size_mb:.1f}MB → {compressed_size_mb:.1f}MB)，使用原始文件")
                        # 删除压缩后的文件
                        try:
                            os.remove(compressed_path)
                        except:
                            pass
            except Exception as e:
                logger.warning(f"[Gemini Video] 视频压缩失败，尝试使用原始文件: {e}")

            
            # 第二步：根据上传模式选择分析方式
            upload_mode = self.config.get("upload_mode", "base64")
            
            # 自动模式：根据文件大小选择最优上传方式
            if upload_mode == "auto":
                if file_size_mb < 10:
                    upload_mode = "base64"
                    logger.info(f"[Gemini Video] 自动模式: 视频大小 {file_size_mb:.1f}MB < 10MB，选择 base64 模式")
                else:
                    upload_mode = "file_api"
                    logger.info(f"[Gemini Video] 自动模式: 视频大小 {file_size_mb:.1f}MB >= 10MB，选择 file_api 模式")
            
            api_config = await self._get_api_config()
            gemini_analysis_result = ""
            
            # 使用信号量限制并发数
            async with self.concurrency_limiter:
                if upload_mode == "file_api":
                    # 文件上传 API 模式：上传到服务器，使用返回的 CDN URL
                    logger.info(f"[Gemini Video] Using File Upload API mode")
                    try:
                        # 1. 上传文件到 /v1/files
                        file_info = await self._upload_file_to_api(local_path, api_config)
                        logger.info(f"[Gemini Video] File uploaded successfully")
                        
                        # 2. 使用返回的信息进行分析（优先使用 CDN URL）
                        async for result_text in self._call_gemini_api_with_file_id(file_info, prompt or "Describe this video."):
                            gemini_analysis_result += result_text
                        
                        if not gemini_analysis_result:
                            return f"❌ 视频分析失败。API 未返回有效结果。"
                            
                        logger.info("[Gemini Video] File API flow analysis success.")
                    except Exception as e:
                        logger.error(f"[Gemini Video] File API mode failed: {e}", exc_info=True)
                        return f"❌ 视频分析失败: {str(e)}"
                else:
                    # Base64 编码模式（默认）
                    max_size_mb = self.config.get("max_base64_size_mb", 30)  # Base64 模式建议最大文件大小
                    if file_size_mb > max_size_mb:
                        return f"❌ 视频文件过大 ({file_size_mb:.1f}MB)，Base64 模式最大支持 {max_size_mb}MB。如需上传更大文件，请将 upload_mode 设置为 file_api。"
                    
                    try:
                        logger.info(f"[Gemini Video] Using Base64 encoding mode")
                        
                        # 这是一个耗时 CPU 操作，对于大文件会阻塞事件循环，必须放入线程池执行
                        def _read_and_encode(path):
                            with open(path, "rb") as video_file:
                                return base64.b64encode(video_file.read()).decode("utf-8")
                        
                        logger.info(f"[Gemini Video] Encoding video to Base64 (in thread pool)...")
                        b64_data = await asyncio.to_thread(_read_and_encode, local_path)
                        
                        data_uri = f"data:video/mp4;base64,{b64_data}"
                        logger.info(f"[Gemini Video] Calling Gemini API with Base64...")
                        
                        async for result_text in self._call_gemini_api_stream(data_uri, prompt or "Describe this video."):
                            gemini_analysis_result += result_text
                        
                        if not gemini_analysis_result:
                            return f"❌ 视频分析失败。API 未返回有效结果。"
                            
                        logger.info("[Gemini Video] Base64 flow analysis success.")
                    except Exception as e:
                        logger.error(f"[Gemini Video] Base64 mode failed: {e}", exc_info=True)
                        return f"❌ 视频分析失败: {str(e)}"
                
                logger.info(f"[Gemini Video] Analysis complete, length: {len(gemini_analysis_result)}")
                
                # 存入缓存 (仅默认提示词的情形)
                if is_default_prompt:
                    self.video_analysis_cache[local_path] = gemini_analysis_result

                # 清理临时文件
                if is_temp:
                    try:
                        os.remove(local_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {local_path}: {e}")

                return gemini_analysis_result

        except Exception as e:
            logger.error(f"[Gemini Video] Analysis error: {e}", exc_info=True)
            raise e


    async def _download_from_url_with_retry(self, url: str, target_path: str, max_retries: int | None = None) -> str:
        """从 URL 下载文件，支持重试、超时控制和下载速度监控。包含此 URL 的并发锁。"""
        
        # 检查是否已有相同 URL 正在下载
        if url in self._downloading_urls:
            logger.info(f"[Gemini Video] URL 正在下载中，等待合并请求: {url}")
            # 简单的自旋等待，直到它从集合中移除
            for _ in range(60): # 最多等 60 秒
                if url not in self._downloading_urls:
                    # 下载完成（假定成功），直接返回
                    if os.path.exists(target_path):
                         logger.info(f"[Gemini Video] 检测到并发下载已完成，直接复用: {target_path}")
                         return target_path
                    break # 如果不存在，说明之前的失败了，重新下载
                await asyncio.sleep(1)
        
        self._downloading_urls.add(url)
        try:
            return await self._internal_download_from_url(url, target_path, max_retries)
        finally:
            self._downloading_urls.discard(url)

    async def _internal_download_from_url(self, url: str, target_path: str, max_retries: int | None = None) -> str:
        """实际执行下载逻辑的内部函数"""
        # 默认 300秒，作为最后的安全底线防止死锁
        safe_read_timeout = self.config.get("download_stream_timeout", 300)
        actual_max_retries = max_retries if max_retries is not None else self.config.get("download_retries", 3)
        retry_delay = self.config.get("download_retry_delay", 5)
        proxy = self.config.get("proxy", "")
        
        # 下载监控配置
        stall_check_interval = 10  # 每10秒检查一次是否停滞

        for i in range(actual_max_retries):
            try:
                if i > 0:
                    logger.info(f"[Gemini Video] 等待 {retry_delay} 秒后进行下一次重试...")
                    await asyncio.sleep(retry_delay)

                logger.info(f"[Gemini Video] 下载文件 (第 {i+1}/{actual_max_retries} 次): {url}")
                
                # 构造请求头
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                
                # 仅对腾讯系域名添加 Referer，避免对其他网站造成防盗链问题
                parsed_url = urlparse(url)
                hostname = parsed_url.hostname or ""
                if hostname.endswith((".qq.com", ".qq.com.cn", ".tencent.com")):
                    headers["Referer"] = "https://www.qq.com/"
                    logger.debug(f"[Gemini Video] 检测到腾讯域名，添加 QQ Referer")
                
                # 创建带代理配置的客户端
                client_kwargs = {
                    "timeout": httpx.Timeout(safe_read_timeout, connect=10.0),
                    "headers": headers,
                    "follow_redirects": True
                }
                
                # 智能代理逻辑：如果是国内域名，强制不走代理
                if hostname and (hostname.endswith(".qq.com") or hostname.endswith(".qq.com.cn") or hostname.endswith(".tencent.com")):
                    logger.info(f"[Gemini Video] 检测到国内域名 ({hostname})，强制直连 (跳过代理)")
                elif proxy:
                    client_kwargs["proxy"] = proxy

                async with httpx.AsyncClient(**client_kwargs) as client:
                    async with client.stream('GET', url) as response:
                        response.raise_for_status()
                        logger.info(f"[Gemini Video] 连接建立成功，开始接收数据...")
                        
                        # 初始化监控变量
                        downloaded_bytes = 0
                        last_check_time = time.time()
                        last_check_bytes = 0
                        
                        # 使用异步文件写入，避免阻塞事件循环
                        async with aiofiles.open(target_path, 'wb') as f:
                            async for chunk in response.aiter_bytes():
                                await f.write(chunk)  # 异步写入
                                downloaded_bytes += len(chunk)
                                
                                # 检查下载速度/停滞
                                current_time = time.time()
                                elapsed = current_time - last_check_time
                                
                                if elapsed >= stall_check_interval:
                                    # 计算这段时间的平均速度
                                    bytes_since_last_check = downloaded_bytes - last_check_bytes
                                    speed_kb_per_sec = (bytes_since_last_check / 1024) / elapsed
                                    
                                    logger.info(f"[Gemini Video] 下载进度: {downloaded_bytes / 1024 / 1024:.2f} MB (速度: {speed_kb_per_sec:.2f} KB/s)")
                                    
                                    # 停滞检测：如果这段时间内没有任何数据写入（且不是还没开始）
                                    if bytes_since_last_check == 0:
                                         raise Exception(
                                            f"下载停滞: 在 {elapsed:.1f} 秒内未接收到任何数据"
                                        )
                                    
                                    # 更新检查点
                                    last_check_time = current_time
                                    last_check_bytes = downloaded_bytes
                        
                        logger.info(f"[Gemini Video] 下载完成: {downloaded_bytes / 1024 / 1024:.2f} MB")
                return target_path
                
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(f"[Gemini Video] 下载网络错误 (第 {i+1} 次) [{type(e).__name__}]: {e}")
                # 清理可能存在的不完整文件
                if os.path.exists(target_path):
                    try:
                        os.remove(target_path)
                        logger.debug(f"[Gemini Video] 已清理不完整文件: {target_path}")
                    except:
                        pass
                if i == actual_max_retries - 1:
                    raise e
            except Exception as e:
                logger.warning(f"[Gemini Video] 下载遇到异常 (第 {i+1} 次) [{type(e).__name__}]: {e}")
                # 清理可能存在的不完整文件
                if os.path.exists(target_path):
                    try:
                        os.remove(target_path)
                        logger.debug(f"[Gemini Video] 已清理不完整文件: {target_path}")
                    except:
                        pass
                if i == actual_max_retries - 1:
                    raise e
        
        raise Exception("下载失败，超过最大重试次数")

    async def _store_video(self, source_path: str) -> str:
        """将视频移动或复制到插件存储目录"""
        if not self.video_storage_path:
             # 如果没有配置存储目录，直接返回源路径（还在临时目录）
            return source_path
            
        file_name = f"video_{os.path.basename(source_path)}"
        # 如果文件名没有时间戳，加上防止重名
        if "video_" not in file_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"video_{timestamp}.mp4"

        target_path = self.video_storage_path / file_name
        
        try:
            # 使用 asyncio.to_thread 避免阻塞事件循环
            await asyncio.to_thread(shutil.copy2, source_path, target_path)
            logger.info(f"[Gemini Video] 视频已复制到存储目录: {target_path}")
            return str(target_path)
        except Exception as e:
            logger.error(f"[Gemini Video] 存储视频失败: {e}")
            return source_path

    async def _download_video(self, video: Video, event: AstrMessageEvent) -> str:
        """下载视频到本地存储目录，尝试多种策略（带超时和重试）"""
        
        # 1. 尝试直接获取本地路径 (仅当它本来就在一个外部存在的路径时使用)
        potential_paths = []
        if video.path: potential_paths.append(video.path)
        
        for p in potential_paths:
            if p and os.path.isabs(p) and os.path.exists(p) and os.path.isfile(p):
                # 如果这个路径就在我们的存储目录里，我们要忽略它以强制下载
                if self.video_storage_path and str(self.video_storage_path) in p:
                    continue
                # 跳过临时目录中的文件
                if "temp" in p.lower():
                    continue

                logger.info(f"[Gemini Video] 发现外部本地视频文件: {p}")
                return await self._store_video(p)
        
        # 2. 尝试使用 video.convert_to_file_path() (AstrBot 内置转换)
        try:
            path = await video.convert_to_file_path()
            if path and os.path.exists(path):
                logger.info(f"[Gemini Video] AstrBot 转换路径成功: {path}")
                return await self._store_video(path)
        except Exception:
            pass

        # 3. 尝试 URL 下载 (标准流程)
        url = getattr(video, "url", None) or video.file
        if url and url.startswith("http"):
            # 直接使用带进度的下载方法（移除不稳定的 OneBot download_file API）
            try:
                logger.info(f"[Gemini Video] 开始下载视频: {url}")
                data_dir = StarTools.get_data_dir(self.name)
                download_dir = data_dir / "temp"
                download_dir.mkdir(parents=True, exist_ok=True)
                video_file_path = str(download_dir / f"{uuid.uuid4().hex}.mp4")
                # 使用带重试和进度显示的下载方法
                path = await self._download_from_url_with_retry(url, video_file_path)
                if path and os.path.exists(path):
                    return await self._store_video(path)
            except Exception as e:
                logger.warning(f"[Gemini Video] URL 下载失败: {e}")

        # 4. 尝试 OneBot API (针对 LLOneBot 等) - 带超时和重试
        if event and isinstance(event, AiocqhttpMessageEvent):
            try:
                bot = event.bot
                file_id = getattr(video, "file_id", None) or video.file
                if file_id:
                    # 尝试 get_file (通用) - 添加超时和重试
                    get_file_timeout = self.config.get("get_file_timeout", 30)
                    get_file_retries = self.config.get("get_file_retries", 2)
                    
                    res = None
                    for attempt in range(get_file_retries):
                        try:
                            logger.info(f"[Gemini Video] 尝试 OneBot get_file (file_id={file_id}, 第 {attempt+1}/{get_file_retries} 次)")
                            res = await asyncio.wait_for(
                                bot.call_action("get_file", file_id=file_id),
                                timeout=get_file_timeout
                            )
                            logger.info(f"[Gemini Video] get_file 成功返回")
                            break  # 成功则跳出重试循环
                        except asyncio.TimeoutError:
                            logger.warning(f"[Gemini Video] get_file 超时 (第 {attempt+1}/{get_file_retries} 次, {get_file_timeout}秒)")
                            if attempt == get_file_retries - 1:
                                raise
                            await asyncio.sleep(2)  # 重试前等待2秒
                        except Exception as e:
                            logger.warning(f"[Gemini Video] get_file 失败 (第 {attempt+1}/{get_file_retries} 次): {e}")
                            if attempt == get_file_retries - 1:
                                raise
                            await asyncio.sleep(2)
                    
                    if res:
                        # 有的实现返回 'file' (本地路径) 或 'url'
                        if "file" in res and res["file"] and os.path.exists(res["file"]):
                            path = res["file"]
                            # 检查是否是图片（缩略图）
                            if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                                logger.warning(f"[Gemini Video] get_file 返回了图片路径，可能是缩略图: {path}")
                            else:
                                logger.info(f"[Gemini Video] get_file 返回本地路径: {path}")
                                return await self._store_video(path)
                        
                        if "url" in res and res["url"] and res["url"].startswith("http"):
                            logger.info(f"[Gemini Video] get_file 返回 URL: {res['url']}")
                            url = res["url"]
                            
                            # 直接使用带进度的下载方法
                            data_dir = StarTools.get_data_dir(self.name)
                            download_dir = data_dir / "temp"
                            download_dir.mkdir(parents=True, exist_ok=True)
                            file_name = f"{uuid.uuid4().hex}.mp4"
                            video_file_path = str(download_dir / file_name)
                            
                            try:
                                logger.info(f"[Gemini Video] 开始下载视频（每10秒更新进度）")
                                path = await self._download_from_url_with_retry(url, video_file_path)
                                if path and os.path.exists(path):
                                    return await self._store_video(path)
                            except Exception as e:
                                logger.error(f"[Gemini Video] 下载失败: {e}")

            except Exception as e_ob:
                logger.warning(f"[Gemini Video] OneBot API 获取失败: {e_ob}")

        raise Exception(f"无法下载视频，所有策略均失效。File info: {video}")


    
    async def _upload_file_to_api(self, file_path: str, api_config: dict) -> dict:
        """上传文件到 /v1/files API (带进度监控和防卡死支持)"""
        
        url = f"{api_config['base_url']}/v1/files"
        file_type = mimetypes.guess_type(file_path)[0] or 'video/mp4'
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # 定义文件包装器用于监控进度
        class MonitoringFile:
            def __init__(self, path, total_size):
                self.f = open(path, 'rb')
                self.total_size = total_size
                self.bytes_read = 0
                self.last_read_time = time.time()
                self.last_log_time = 0
                
            def read(self, size=-1):
                data = self.f.read(size)
                if data:
                    self.bytes_read += len(data)
                    self.last_read_time = time.time()
                    
                    # 每 2 秒打印一次日志
                    current_time = time.time()
                    if current_time - self.last_log_time >= 2:
                        progress = (self.bytes_read / self.total_size) * 100
                        speed = (len(data) / 1024 / 1024) # 这里的速度计算不准确，只是瞬时，仅打印进度即可
                        logger.info(f"[Gemini Video] 上传进度: {progress:.1f}% ({self.bytes_read/1024/1024:.1f}/{self.total_size/1024/1024:.1f} MB)")
                        self.last_log_time = current_time
                return data
                
            def close(self):
                self.f.close()

        # 准备上传
        monitor_file = MonitoringFile(file_path, file_size)
        
        # 监控任务：检查上传是否卡死
        async def _stall_monitor():
            while True:
                await asyncio.sleep(5)
                if time.time() - monitor_file.last_read_time > 20: # 20秒无读取则认为卡死
                    if monitor_file.bytes_read < monitor_file.total_size:
                        logger.error("[Gemini Video] 上传检测到卡死 (20秒无数据传输)")
                        # 这里我们无法直接中断 httpx 请求，但抛出异常或取消 task 会在外部处理
                        # 简单的做法是让这个 monitor 抛出 CancelledError 给主任务? 
                        # 由于 httpx 是同步阻塞在这里的 await，我们需要从外部 cancel 它。
                        # 但这里我们在同一个函数里。
                        # 实际上，httpx 的 read 是在 C 层面或者是 loop 中。
                        pass
        
        # 使用 multipart/form-data 上传
        # 为了支持监控，我们需要将 monitor_file 作为文件对象传递
        # 注意：httpx 会在后台线程或事件循环中调用 read()
        
        files = {
            'file': (file_name, monitor_file, file_type)
        }
        
        headers = {
            "Authorization": f"Bearer {api_config['api_key']}"
        }
        
        logger.info(f"[Gemini Video] 开始上传文件: {file_name} ({file_size/1024/1024:.1f} MB)")
        
        # 设置一个较长的安全超时，主要依赖 stall 监控 (这里简化，先设长一点)
        # 如果需要完美的 stall 监控，需要将 client.post 放入 task 并由 monitor 取消
        # 这里为了简单，我们先设置长超时，并依赖 MonitoringFile 的 read 日志来观察
        # 如果真卡死，用户会在 safe_timeout 后得到错误，或者我们可以改进 monitor 逻辑
        
        # 改进方案：使用 custom timeout transport 或者直接长超时。
        # 鉴于 python httpx 的限制，我们设置一个极长的 read timeout (例如 1小时)
        # 但我们用 monitor task 来主动取消请求
        
        safe_upload_timeout = self.config.get("upload_stream_timeout", 3600)
        timeout = httpx.Timeout(float(safe_upload_timeout), connect=30.0) # 读取超时，靠监控任务中断
        proxy = self.config.get("proxy", "")
        
        logger.info(f"[Gemini Video] Upload Configuration - Timeout: {safe_upload_timeout}s, Proxy: {proxy if proxy else 'None'}")
        
        client_kwargs = {"timeout": timeout}
        if proxy: client_kwargs["proxy"] = proxy

        try:
            # 定义上传任务
            # 定义同步上传函数 (将在线程中运行)
            # 关键修复：使用同步客户端 + asyncio.to_thread，确保 MonitorFile.read() 的阻塞只会发生在工作线程中，
            # 而不会阻塞主事件循环，从而避免机器人在上传大文件时无响应。
            def _run_sync_upload():
                # 在线程中构建同步客户端
                sync_client_kwargs = {"timeout": httpx.Timeout(float(safe_upload_timeout), connect=30.0)}
                if proxy: sync_client_kwargs["proxy"] = proxy

                with httpx.Client(**sync_client_kwargs) as client:
                    data = {"purpose": "assistants"}
                    # 注意：httpx 同步客户端会同步调用 monitor_file.read()，但这发生在线程中，是安全的。
                    resp = client.post(url, headers=headers, files=files, data=data)
                    resp.raise_for_status()
                    return resp.json()

            # 定义监控任务
            async def _monitor():
                while True:
                    await asyncio.sleep(5)
                    # 检查是否完成
                    if monitor_file.bytes_read >= monitor_file.total_size:
                        pass 
                    elif time.time() - monitor_file.last_read_time > 30: # 30秒无读取判定为卡死
                         raise TimeoutError("上传卡死：30秒内无数据传输")

            # 重试循环
            max_retries = self.config.get("upload_retries", 3)
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        logger.info(f"[Gemini Video] 上传重试 (第 {attempt+1}/{max_retries} 次)...")
                        # 重置文件指针和计数器
                        monitor_file.f.seek(0)
                        monitor_file.bytes_read = 0
                        monitor_file.last_read_time = time.time()
                        await asyncio.sleep(3) # 稍作等待

                    # 并发执行 (上传在线程池中，监控在主循环中)
                    upload_task = asyncio.create_task(asyncio.to_thread(_run_sync_upload))
                    monitor_task = asyncio.create_task(_monitor())
                    
                    done, pending = await asyncio.wait(
                        [upload_task, monitor_task], 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # 清理
                    for t in pending: t.cancel()
                    
                    # 检查结果
                    if upload_task in done:
                        try:
                            return upload_task.result()
                        except Exception as e:
                            # 捕获上传任务中的异常（如 ReadError）
                            raise e
                    else:
                        # 监控任务先完成（只能是抛出异常）
                        monitor_task.result()
                        
                except Exception as e:
                    if isinstance(e, (httpx.ReadError, httpcore.ReadError)):
                        logger.warning(f"[Gemini Video] 上传遇到 ReadError (可能是连接超时): {e}")
                        if attempt == max_retries - 1:
                            logger.error("[Gemini Video] 提示: 频繁的 ReadError 通常意味着中间节点(代理/网关)限制了连接时长 (常见为120秒)。建议尝试关闭代理(如果使用直连优化域名)或更换节点。")
                    else:
                         logger.warning(f"[Gemini Video] 上传尝试 {attempt+1} 失败: {e}")
                    
                    if attempt == max_retries - 1:
                        raise e
                    # 继续下一次重试

        except Exception as e:
            logger.error(f"[Gemini Video] 上传失败: {e}")
            monitor_file.close() # 确保关闭
            raise e
        finally:
            monitor_file.close()
        

    
    async def _call_gemini_api_with_file_id(self, file_info: dict, prompt: str):
        """使用上传后的文件信息调用 Gemini API 进行分析
        
        Args:
            file_info: 上传文件后返回的信息（包含 id, url 等）
            prompt: 提示词
            
        Yields:
            分析结果文本片段
        """
        api_config = await self._get_api_config()
        
        # 优先使用返回的 URL（CDN URL），如果没有则使用 file_id
        file_url = file_info.get("url")
        
        if file_url:
            # 如果有 URL，直接使用 URL 分析（类似现有的 URL 分析）
            logger.info(f"[Gemini Video] Using uploaded file URL: {file_url}")
            async for chunk in self._call_gemini_api_stream(file_url, prompt):
                yield chunk
        else:
            # 如果没有 URL，说明 API 不支持此模式
            raise ValueError("File upload did not return a usable URL. File API mode may not be supported.")



    async def _call_gemini_api_stream(
        self, video_url: str, prompt: str
    ):
        """调用 OpenAI 兼容 API（流式，使用 URL）
        
        Args:
            video_url: 视频 URL
            prompt: 提示词
            
        Yields:
            分析结果文本片段
        """
        # 获取 API 配置
        api_config = await self._get_api_config()
        
        # 构建请求
        payload = {
            "model": api_config["model"],
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": video_url},
                        },
                    ],
                }
            ],
            "max_tokens": self.config.get("max_tokens", 4000),
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_config['api_key']}",
        }

        # 发送流式请求
        async with self.client.stream(
            "POST",
            f"{api_config['base_url']}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                    
                data_str = line[6:]  # 移除 "data: " 前缀
                if data_str.strip() == "[DONE]":
                    break
                    
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        yield content
                            
                except json.JSONDecodeError:
                    continue

    async def _get_api_config(self) -> dict:
        """获取 API 配置"""
        if not self.config:
            self.config = self.context.get_config()
        
        base_url = self.config.get("base_url", "")
        api_key = self.config.get("api_key", "")
        model = self.config.get("model", "gemini-2.5-flash")

        if model == "自定义模型":
            model = self.config.get("custom_model", "")

        state_msg = f"base_url={base_url}, api_key={'********' if api_key else 'None'}"
        logger.debug(f"[Gemini Video] API Config - {state_msg}")

        if not api_key:
             raise ValueError(f"未配置 API Base URL 或 API Key。状态: {state_msg}")
        
        # 默认 Base URL
        if not base_url:
            base_url = "https://generativelanguage.googleapis.com"
        
        # 去除末尾斜杠
        base_url = base_url.rstrip("/")
        
        return {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }

    async def _compress_video_if_needed(self, input_path: str) -> str:
        """如果视频超过阈值，调用 ffmpeg 进行压缩"""
        if not self.config.get("enable_compression", True):
            return input_path
            
        threshold_mb = self.config.get("compression_threshold_mb", 25)
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        
        if file_size_mb <= threshold_mb:
            return input_path
            
        # 检查 ffmpeg 是否可用
        if not shutil.which("ffmpeg"):
            logger.warning("[Gemini Video] 未找到 ffmpeg，跳过压缩。建议安装 ffmpeg 以优化大文件上传。")
            return input_path
            
        logger.info(f"[Gemini Video] 视频大小 ({file_size_mb:.1f} MB) 超过阈值 ({threshold_mb} MB)，开始压缩...")
        
        # 构造输出文件名
        input_file = Path(input_path)
        output_file = input_file.parent / f"{input_file.stem}_compressed.mp4"
        
        try:
            # 快速压缩参数（平衡速度和效果）
            # - 使用 libx264 (H.264) - 编码速度比 H.265 快很多
            # - crf 26 稍低的值，加快编码速度
            # - preset veryfast 快速编码
            # - scale=-2:720 降低分辨率到720p
            # - 音频使用 aac 编码器，码率 128k
            cmd = [
                "ffmpeg", "-y", 
                "-i", input_path,
                "-vf", "scale=-2:720",    # 720p，宽度自动适配
                "-c:v", "libx264",        # 使用 H.264 编码器（速度快）
                "-preset", "veryfast",    # 快速编码
                "-crf", "26",             # CRF 26 - 平衡质量和速度
                "-c:a", "aac",            # 音频使用 AAC
                "-b:a", "128k",           # 音频码率 128kbps
                str(output_file)
            ]
            
            logger.info(f"[Gemini Video] FFmpeg command: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 等待压缩完成
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"[Gemini Video] FFmpeg compression failed: {stderr.decode()}")
                return input_path
            
            new_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            logger.info(f"[Gemini Video] 压缩完成: {file_size_mb:.1f} MB -> {new_size_mb:.1f} MB")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"[Gemini Video] Error during compression: {e}")
            if output_file.exists():
                try: output_file.unlink()
                except: pass
            return input_path
