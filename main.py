from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import *
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
from astrbot.core.utils.io import download_file
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.agent.run_context import ContextWrapper
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
import httpx
import os
import json
import asyncio
import shutil
import time
from pathlib import Path
from typing import Optional, Any
import uuid

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
        
        # If no URL provided in args, try to find one in the current event context
        if not video_url:
            video_comp = await self.plugin._find_video_component(context.context.event)
            if video_comp:
                video_url = video_comp.file
                logger.info(f"[Gemini Video] Tool auto-detected video in context: {video_url}")
            else:
                return "Please provide a video URL or send a video first."

        try:
             logger.info(f"[Gemini Video] Tool call started. URL: {video_url}, Prompt: {prompt}")
             
             # 发送中间提示 (使用更安全的方式)
             try:
                 if context.context and context.context.event:
                      watching_hint = self.plugin.config.get("watching_hint", "亚托莉正在看视频哦~")
                      await self.plugin.context.send_message(context.context.event.unified_msg_origin, MessageChain([Plain(watching_hint)]))
                 else:
                     logger.warning("[Gemini Video] context.context.event is None in tool call.")
             except Exception as e_hint:
                 logger.warning(f"[Gemini Video] Failed to send hint: {e_hint}")

             # Pass the event context to the analysis method for more robust downloading
             logger.info(f"[Gemini Video] Calling _perform_video_analysis...")
             result = await self.plugin._perform_video_analysis(video_url, prompt, event=context.context.event)
             logger.info(f"[Gemini Video] _perform_video_analysis returned. Length: {len(result) if result else 0}")
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
        
        # 缓存与同步
        self.video_analysis_cache: dict[str, str] = {} # msg_id/file_path -> analysis
        self.analysis_lock = asyncio.Lock()

        # 创建视频存储目录
        storage_path = self.config.get("video_storage_path", "videos")
        if storage_path:
            # 相对路径相对于插件目录
            plugin_dir = Path(__file__).parent
            self.video_storage_path = plugin_dir / storage_path
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
        # 启动清理任务
        asyncio.create_task(self._cleanup_loop())
        
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
                        keys_to_del = [k for k, v in self.video_cache.items() if v == str(item)]
                        for k in keys_to_del: self.video_cache.pop(k, None)
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
                # 1. 下载/复制视频到缓存
                local_path = await self._download_video(video_comp, event)
                if local_path:
                    msg_id = str(event.message_obj.message_id)
                    self.video_cache[msg_id] = local_path
                    logger.debug(f"[Gemini Video] 已缓存视频消息 {msg_id} -> {local_path}")
                    
                    # 2. 检查是否需要预解析 (被提及且包含视频)
                    is_to_bot = False
                    # 检查 At
                    for c in event.message_obj.message:
                        if isinstance(c, At) and c.qq == str(event.self_id):
                            is_to_bot = True; break
                    
                    # 检查名字
                    plain_text = event.message_str
                    bot_names = ["亚托莉", "萝卜子", "ATRI", "Atri"]
                    if not is_to_bot:
                        for name in bot_names:
                            if name in plain_text:
                                is_to_bot = True; break
                    
                    # 仅下载缓存，不再主动预解析（节省 API 消耗）
                    logger.debug(f"[Gemini Video] Video cached and ready for tool call: {local_path}")

            except Exception as e:
                logger.warning(f"[Gemini Video] 自动缓存解析视频失败: {e}")

    async def _find_video_component(self, event: AstrMessageEvent) -> Video | None:
        """从消息或引用中查找视频组件"""
        if not event or not event.message_obj or not event.message_obj.message:
            return None
            
        # 1. 检查引用消息
        for comp in event.message_obj.message:
            if isinstance(comp, Reply):
                # 尝试从缓存获取
                if str(comp.id) in self.video_cache:
                    local_path = self.video_cache[str(comp.id)]
                    if os.path.exists(local_path):
                        return Video(file=local_path, path=local_path)
                
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
            
            # 调用主模型进行生成
            try:
                # 获取当前会话使用的 LLM Provider ID
                provider_id = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
                
                # 获取当前会话的人格设置
                personality = await self.context.persona_manager.get_default_persona_v3(event.unified_msg_origin)
                system_prompt = personality['prompt']
                # begin_dialogs 用于设定语气
                contexts = personality['_begin_dialogs_processed']

                # 处理空 Prompt
                final_user_prompt = prompt if prompt.strip() else "Look at this video."

                # 构建给主模型的 Prompt - Double Injection Strategy
                final_prompt = (
                    f"[System Instruction: You are {system_prompt}]\n"
                    f"[Context: The user sent a video. Here is a description of the video content:]\n\n"
                    f"{gemini_analysis_result}\n\n"
                    f"[User Request: {final_user_prompt}]\n\n"
                    f"[Task: Reply to the User Request based on the video description. Important: You must ACT AS your persona defined in System Instruction. Do NOT act as an AI assistant. Stay in character.]"
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
        """执行视频分析的核心逻辑"""
        try:
            # 检查是否启用 URL 直接分析
            use_url_analysis = self.config.get("use_url_analysis", True)
            
            # 如果是远程 URL 且启用了直接分析，则跳过下载
            if use_url_analysis and (video_url.startswith("http://") or video_url.startswith("https://")):
                logger.info(f"[Gemini Video] Using direct URL analysis for: {video_url}")
                try:
                    gemini_analysis_result = ""
                    async for chunk in self._call_gemini_api_stream(video_url, prompt or "Describe this video."):
                        gemini_analysis_result += chunk
                    
                    if not gemini_analysis_result:
                        return "❌ 视频分析失败。API 未返回有效结果。"
                    
                    logger.info("[Gemini Video] URL analysis success.")
                    return gemini_analysis_result
                except Exception as e:
                    logger.warning(f"[Gemini Video] URL analysis failed: {e}, falling back to download mode.")
                    # 如果 URL 分析失败，继续使用下载模式
            
            # 1. 处理视频路径/URL，并下载到本地
            local_path = ""
            is_temp = False

            if video_url.startswith("file:///"):
                local_path = video_url[8:]
            elif os.path.exists(video_url):
                local_path = video_url
            else:
                # 优先尝试使用 _download_video 获取（通过 OneBot API 或多种策略）
                try:
                    dummy_video = Video(file=video_url)
                    stored_path = await self._download_video(dummy_video, event)
                    if stored_path and os.path.exists(stored_path):
                        local_path = stored_path
                        is_temp = False 
                except Exception as e_dl:
                    logger.warning(f"[Gemini Video] _download_video failed: {e_dl}, trying fallback direct download.")
                    temp_video_path = f"{uuid.uuid4()}.mp4"
                    local_path = os.path.join(get_astrbot_data_path(), temp_video_path)
                    logger.info(f"[Gemini Video] Fallback downloading video from {video_url} to {local_path}")
                    await self._download_from_url_with_retry(video_url, local_path)
                    is_temp = True
            
            if not local_path or not os.path.exists(local_path):
                return "❌ Video file not found or download failed."

            # 检查文件大小
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            max_size = self.config.get("max_video_size_mb", 100)
            if file_size_mb > max_size:
                return f"❌ 视频文件过大 ({file_size_mb:.1f}MB)，最大支持 {max_size}MB。"

            # 2. 不再检查分析结果缓存，实现强制重新分析
            is_default_prompt = not prompt or prompt == "Describe this video." or "分析" in prompt
            logger.info(f"[Gemini Video] Video ready at {local_path}, checking API mode...")
            
            # 使用锁防止并发分析同一视频
            async with self.analysis_lock:
                api_config = await self._get_api_config()
                gemini_analysis_result = ""
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                
                # 使用 Base64 编码上传
                max_size_mb = 30  # Base64 模式建议最大文件大小
                if file_size_mb > max_size_mb:
                    return f"❌ 视频文件过大 ({file_size_mb:.1f}MB)，Base64 模式最大支持 {max_size_mb}MB。"
                
                try:
                    logger.info(f"[Gemini Video] Using Base64 flow. Size: {file_size_mb:.1f}MB")
                    import base64
                    with open(local_path, "rb") as video_file:
                        b64_data = base64.b64encode(video_file.read()).decode("utf-8")
                    
                    data_uri = f"data:video/mp4;base64,{b64_data}"
                    logger.info(f"[Gemini Video] Calling OpenAI compatible API with Base64...")
                    
                    async for result_text in self._call_gemini_api_stream(data_uri, prompt or "Describe this video."):
                        gemini_analysis_result += result_text
                    
                    if not gemini_analysis_result:
                        return f"❌ 视频分析失败。API 未返回有效结果。"
                        
                    logger.info("[Gemini Video] Base64 flow analysis success.")
                except Exception as e:
                    logger.error(f"[Gemini Video] Analysis failed: {e}", exc_info=True)
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
        """从 URL 下载文件，支持重试和超时控制"""
        read_timeout = self.config.get("download_timeout", 20)
        actual_max_retries = max_retries if max_retries is not None else self.config.get("download_retries", 3)
        retry_delay = self.config.get("download_retry_delay", 5)
        proxy = self.config.get("proxy", "")

        for i in range(actual_max_retries):
            try:
                if i > 0:
                    logger.info(f"[Gemini Video] 等待 {retry_delay} 秒后进行下一次重试...")
                    await asyncio.sleep(retry_delay)

                logger.info(f"[Gemini Video] 下载文件 (第 {i+1}/{actual_max_retries} 次): {url} (Timeout: {read_timeout}s)")
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": "https://www.qq.com/"
                }
                
                # 创建带代理配置的客户端
                client_kwargs = {
                    "timeout": httpx.Timeout(read_timeout, connect=10.0),
                    "headers": headers,
                    "follow_redirects": True
                }
                if proxy:
                    client_kwargs["proxy"] = proxy

                async with httpx.AsyncClient(**client_kwargs) as client:
                    async with client.stream('GET', url) as response:
                        response.raise_for_status()
                        with open(target_path, 'wb') as f:
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                return target_path
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(f"[Gemini Video] 下载失败 (第 {i+1} 次): {e}")
                if i == max_retries - 1:
                    raise e
                # 立即重试，不等待
            except Exception as e:
                if i == max_retries - 1:
                    raise e
                logger.warning(f"[Gemini Video] 下载遇到错误: {e}, 即将重试")
                # 立即重试，不等待
        
        raise Exception("下载失败，超过最大重试次数")

    async def _download_video(self, video: Video, event: AstrMessageEvent) -> str:
        """下载视频到本地存储目录，尝试多种策略 (强制重新下载)"""
        
        # 1. 尝试直接获取本地路径 (仅当它本来就在一个外部存在的路径时使用)
        # 不再检查我们自己的 temp 或 videos 目录，以实现强制下载
        potential_paths = []
        if video.path: potential_paths.append(video.path)
        # 注意：不再将 video.file 直接视为本地路径，除非它是绝对路径且不在我们的存储区
        
        for p in potential_paths:
            if p and os.path.isabs(p) and os.path.exists(p) and os.path.isfile(p):
                # 如果这个路径就在我们的存储目录里，我们要忽略它以强制下载
                if self.video_storage_path and str(self.video_storage_path) in p:
                    continue
                if "temp" in p and get_astrbot_data_path() in p:
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
            # 优先尝试 OneBot download_file API (如果可用)
            if event and isinstance(event, AiocqhttpMessageEvent):
                try:
                    logger.info(f"[Gemini Video] 尝试 OneBot download_file API: {url}")
                    file_name = f"{uuid.uuid4().hex}.mp4"
                    res = await event.bot.call_action("download_file", url=url, name=file_name)
                    if res and isinstance(res, dict) and "file" in res:
                        path = res["file"]
                        if path and os.path.exists(path):
                            logger.info(f"[Gemini Video] OneBot download_file 成功: {path}")
                            return await self._store_video(path)
                except Exception as e:
                    logger.warning(f"[Gemini Video] OneBot download_file API 失败: {e}")

            try:
                logger.info(f"[Gemini Video] 尝试本地下载: {url}")
                download_dir = os.path.join(get_astrbot_data_path(), "temp")
                video_file_path = os.path.join(download_dir, f"{uuid.uuid4().hex}.mp4")
                # 使用带重试的下载方法
                path = await self._download_from_url_with_retry(url, video_file_path)
                if path and os.path.exists(path):
                    return await self._store_video(path)
            except Exception as e:
                logger.warning(f"[Gemini Video] URL 下载失败: {e}")

        # 4. 尝试 OneBot API (针对 LLOneBot 等)
        # LLOneBot 的视频往往在 QQ 的临时目录，AstrBot 可能拿不到权限或路径不对
        # 需要调用 get_group_file_url 或 get_file 获取真实路径/链接
        if event and isinstance(event, AiocqhttpMessageEvent):
            try:
                bot = event.bot
                file_id = getattr(video, "file_id", None) or video.file
                if file_id:
                    # 尝试 get_file (通用)
                    try:
                        logger.info(f"[Gemini Video] 尝试 OneBot get_file (file_id={file_id})")
                        res = await bot.call_action("get_file", file_id=file_id)
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
                                # 尝试 OneBot download_file API
                                file_name = f"{uuid.uuid4().hex}.mp4"
                                try:
                                    dl_res = await bot.call_action("download_file", url=url, name=file_name)
                                    if dl_res and isinstance(dl_res, dict) and "file" in dl_res:
                                        path = dl_res["file"]
                                        if path and os.path.exists(path):
                                            return await self._store_video(path)
                                except Exception as e:
                                    logger.warning(f"[Gemini Video] OneBot download_file API 失败: {e}")

                                # 降级到本地下载
                                download_dir = os.path.join(get_astrbot_data_path(), "temp")
                                video_file_path = os.path.join(download_dir, file_name)
                                # 使用带重试的下载方法
                                path = await self._download_from_url_with_retry(res["url"], video_file_path)
                                if path: return await self._store_video(path)
                    except Exception as e:
                        logger.debug(f"[Gemini Video] get_file 失败: {e}")

            except Exception as e_ob:
                logger.warning(f"[Gemini Video] OneBot API 获取失败: {e_ob}")

        raise Exception(f"无法下载视频，所有策略均失效。File info: {video}")

    async def _store_video(self, source_path: str) -> str:
        """将视频移动或复制到插件存储目录"""
        if not self.video_storage_path:
             # 如果没有配置存储目录，直接返回源路径（还在临时目录）
            return source_path
            
        file_name = f"video_{os.path.basename(source_path)}"
        # 如果文件名没有时间戳，加上防止重名
        if "video_" not in file_name:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"video_{timestamp}.mp4"

        target_path = self.video_storage_path / file_name
        
        try:
            shutil.copy2(source_path, target_path)
            logger.info(f"[Gemini Video] 视频已复制到存储目录: {target_path}")
            return str(target_path)
        except Exception as e:
            logger.error(f"[Gemini Video] 存储视频失败: {e}")
            return source_path

    async def _upload_file(self, local_path: str, mime_type: str, api_config: dict) -> str:
        """上传文件到 Gemini File API (Resumable Upload)"""
        file_size = os.path.getsize(local_path)
        display_name = os.path.basename(local_path)
        
        # 1. 构造上传 URL
        base_url = api_config["base_url"]
        if "/v1" in base_url:
            base_url = base_url.split("/v1")[0]
            
        upload_url = f"{base_url}/upload/v1beta/files"
        
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
            "x-goog-api-key": api_config["api_key"],
        }
        
        # 2. 初始请求获取上传 session URL
        meta_payload = {"file": {"display_name": display_name}}
        logger.debug(f"[Gemini Video] 开始上传: {upload_url}")
        
        resp_start = await self.client.post(
            upload_url, 
            headers=headers, 
            json=meta_payload
        )
        resp_start.raise_for_status()
        
        session_uri = resp_start.headers.get("x-goog-upload-url") or resp_start.headers.get("X-Goog-Upload-Url")
        if not session_uri:
            logger.error(f"[Gemini Video] 获取 Session URL 失败. 状态码: {resp_start.status_code}, 响应: {resp_start.text[:200]}")
            raise Exception(f"服务商不支持 File API (无 Session URL). 响应: {resp_start.text[:50]}")
            
        # 3. 上传文件内容
        logger.info(f"[Gemini Video] 上传文件内容 ({file_size} bytes)")
        with open(local_path, "rb") as f:
            file_content = f.read()
            
        headers_upload = {
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "x-goog-api-key": api_config["api_key"],
        }
        
        resp_upload = await self.client.post(
            session_uri,
            headers=headers_upload,
            content=file_content
        )
        resp_upload.raise_for_status()
        
        file_info = resp_upload.json()
        file_uri = file_info["file"]["uri"]
        file_name = file_info["file"]["name"]
        logger.info(f"[Gemini Video] 上传成功. URI: {file_uri}, Name: {file_name}")
        
        # 4. 等待文件处理完成
        await self._wait_for_file_active(file_name, api_config)
        
        return file_uri

    async def _wait_for_file_active(self, file_name: str, api_config: dict):
        """等待文件状态变为 ACTIVE"""
        base_url = api_config["base_url"]
        if "/v1" in base_url:
            base_url = base_url.split("/v1")[0]
            
        check_url = f"{base_url}/v1beta/{file_name}"
        headers = {"x-goog-api-key": api_config["api_key"]}
        
        logger.info("[Gemini Video] 等待视频处理...")
        for _ in range(30): # 最多等待 60s
            resp = await self.client.get(check_url, headers=headers)
            resp.raise_for_status()
            info = resp.json()
            state = info.get("state")
            
            if state == "ACTIVE":
                logger.info("[Gemini Video] 视频处理完毕 (ACTIVE)")
                return
            elif state == "FAILED":
                raise Exception(f"视频处理失败: {info}")
                
            await asyncio.sleep(2)
        
        raise Exception("视频处理超时")

    async def _generate_content_stream(self, file_uri: str, prompt: str):
        """调用 Native Gemini API 生成内容 (流式)"""
        api_config = await self._get_api_config()
        
        base_url = api_config["base_url"]
        if "/v1" in base_url:
            base_url = base_url.split("/v1")[0]
            
        model = api_config["model"]
        url = f"{base_url}/v1beta/models/{model}:streamGenerateContent"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}}
                ]
            }],
            "generationConfig": {
                "maxOutputTokens": self.config.get("max_tokens", 4000)
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_config["api_key"]
        }
        
        async with self.client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                err_text = await response.aread()
                logger.error(f"[Gemini Video] API Error {response.status_code}: {err_text}")
                raise Exception(f"API 请求失败: {response.status_code}")

            buffer = ""
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                
                json_str = line[6:].strip()
                if not json_str: continue

                try:
                    chunk = json.loads(json_str)
                    candidates = chunk.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            if text:
                                buffer += text
                                if len(buffer) > 50:
                                    yield buffer
                                    buffer = ""
                except Exception:
                    pass
            
            if buffer:
                yield buffer

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
