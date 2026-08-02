"""
=============================================================================
阿里云百炼 LLM 模块

通过阿里云百炼 API 调用通义千问系列大语言模型（Qwen），支持流式输出、
上下文对话、以及自定义生成参数。

API 兼容 OpenAI 格式：
    POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
=============================================================================

使用方法:
    from src.llm import BailianLLM

    llm = BailianLLM()
    response = llm.generate("你好，请介绍一下自己")
    print(response)

    # 流式输出
    for chunk in llm.stream("讲个故事"):
        print(chunk, end="")
"""

import json
import time
from typing import Any, Generator

import httpx

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BailianLLMError(Exception):
    """LLM 调用过程中的异常"""
    pass


class BailianLLM:
    """
    阿里云百炼大语言模型封装。

    支持:
        - 普通文本生成
        - 流式文本生成
        - 多轮对话
        - 自定义系统提示词
        - 自定义生成参数（温度、最大Token等）
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            api_key:      百炼 API 密钥（默认从全局配置读取）
            model:        模型名称（默认从全局配置读取）
            base_url:     API 基础地址（默认从全局配置读取）
            temperature:  生成温度 (0~2)，越低越确定，越高越随机
            max_tokens:   最大生成 Token 数
            top_p:        Nucleus 采样参数
            max_retries:  失败重试次数
            retry_delay:  重试间隔（秒）
        """
        self.api_key = api_key or settings.BAILIAN_API_KEY
        self.model = model or settings.LLM_MODEL_NAME
        self.base_url = base_url or settings.BAILIAN_API_BASE
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 构造 Chat API 端点
        self._chat_url = f"{self.base_url.rstrip('/')}/chat/completions"

        logger.info(
            f"LLM 初始化: model={self.model}, "
            f"temperature={temperature}, max_tokens={max_tokens}"
        )

    # ================================================================
    # 公开接口
    # ================================================================

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> str:
        """
        生成文本（非流式）。

        Args:
            prompt:        用户输入提示
            system_prompt: 系统提示词（可选）
            history:       历史对话列表，格式: [{"role": "user"|"assistant", "content": "..."}]
            **kwargs:      可覆盖生成参数（temperature, max_tokens, top_p 等）

        Returns:
            生成的文本内容

        Raises:
            BailianLLMError: API 调用失败
        """
        messages = self._build_messages(prompt, system_prompt, history)
        params = self._get_params(**kwargs)

        response_data = self._call_api(messages, params, stream=False)
        return self._extract_content(response_data)

    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        流式生成文本。

        Args:
            prompt:        用户输入提示
            system_prompt: 系统提示词（可选）
            history:       历史对话列表
            **kwargs:      可覆盖生成参数

        Yields:
            逐个生成的文本片段
        """
        messages = self._build_messages(prompt, system_prompt, history)
        params = self._get_params(**kwargs)

        for chunk in self._call_api_stream(messages, params):
            yield chunk

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> str:
        """
        多轮对话（直接传入消息列表）。

        Args:
            messages: 消息列表，格式:
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."},
                 {"role": "assistant", "content": "..."},
                 {"role": "user", "content": "..."}]
            **kwargs: 可覆盖生成参数

        Returns:
            模型回复文本
        """
        params = self._get_params(**kwargs)
        response_data = self._call_api(messages, params, stream=False)
        return self._extract_content(response_data)

    # ================================================================
    # 内部方法
    # ================================================================

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """构建消息列表"""
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        return messages

    def _get_params(self, **kwargs) -> dict[str, Any]:
        """获取生成参数"""
        return {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
        }

    def _call_api(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any],
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        调用 Chat API。

        Args:
            messages: 消息列表
            params:   生成参数
            stream:   是否为流式调用

        Returns:
            API 响应数据
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": stream,
                    **params,
                }

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                response = httpx.post(
                    self._chat_url,
                    json=payload,
                    headers=headers,
                    timeout=120.0,
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                body = e.response.text
                logger.warning(
                    f"LLM API HTTP 错误 (尝试 {attempt}/{self.max_retries}): "
                    f"status={status}, body={body[:300]}"
                )
                if status in (400, 401, 403):
                    raise BailianLLMError(
                        f"API 请求被拒绝: status={status}, detail={body[:300]}"
                    ) from e
                if status == 429:
                    # 限流，等待更长时间
                    time.sleep(self.retry_delay * attempt * 2)
                    continue

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"LLM API 超时 (尝试 {attempt}/{self.max_retries})"
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM API 未知错误 (尝试 {attempt}/{self.max_retries}): {e}"
                )

            # 重试前等待
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)

        raise BailianLLMError(
            f"LLM API 调用失败（已达最大重试次数 {self.max_retries}）: {last_error}"
        )

    def _call_api_stream(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any],
    ) -> Generator[str, None, None]:
        """
        流式调用 Chat API。

        Yields:
            文本片段
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **params,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            with httpx.stream(
                "POST",
                self._chat_url,
                json=payload,
                headers=headers,
                timeout=120.0,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # 去掉 "data: " 前缀
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"流式调用失败: {e}")
            raise BailianLLMError(f"流式生成失败: {e}") from e

    @staticmethod
    def _extract_content(response_data: dict[str, Any]) -> str:
        """从 API 响应中提取生成的文本"""
        try:
            return response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise BailianLLMError(f"无法从响应中提取内容: {response_data}") from e
