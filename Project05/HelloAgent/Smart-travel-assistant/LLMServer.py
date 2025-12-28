from __future__ import annotations

from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        调用LLMAPI来生成回应（真流式）。
        - 终端会实时打印模型输出
        - 函数最终返回完整字符串（便于后续逻辑继续用）
        """
        print("正在调用大语言模型(流式输出)...")

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )

            parts: List[str] = []
            for event in stream:
                # event 是 ChatCompletionChunk
                if not event.choices:
                    continue

                choice = event.choices[0]
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue

                token = getattr(delta, "content", None)
                if token:
                    print(token, end="", flush=True)
                    parts.append(token)

            print()  # 换行
            answer = "".join(parts)
            print("大语言模型流式响应结束。")
            return answer

        except Exception as e:
            print(f"调用LLMAPI时发生错误:{e}")
            return "错误:调用语言模型服务时出错。"
