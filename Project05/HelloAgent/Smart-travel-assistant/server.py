import os
import re
import ast
from typing import Any, Dict, Tuple

from LLMServer import OpenAICompatibleClient
from wttr import get_weather
from search_attraction import get_attraction


AGENT_SYSTEM_PROMPT = """你是一个严格遵守格式的旅行助手Agent。
你必须只输出一个Thought和一个Action，格式如下：

Thought: <你的思考，简短>
Action: <下面三种之一>
1)get_weather(city="城市名")
2)get_attraction(city="城市名", weather="天气描述")
3)finish(answer="最终给用户的自然语言答案")

注意：
- Action必须在同一行完成。
- 字符串参数可以用单引号或双引号。
- 当你拿到天气与景点信息后，用finish输出最终答案。
"""


available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


def _parse_call(expr: str) -> Tuple[str, Dict[str, Any]]:
    """
    安全解析类似get_weather(city="北京")这种表达式。
    只允许：函数名(纯关键字参数且值为常量)。
    """
    expr = expr.strip()

    m = re.match(r"^([a-zA-Z_]\w*)\s*\((.*)\)\s*$", expr, flags=re.DOTALL)
    if not m:
        raise ValueError(f"无法解析Action表达式:{expr}")

    func_name = m.group(1)
    args_src = m.group(2).strip()

    # 用AST解析关键字参数，避免eval
    node = ast.parse(f"f({args_src})", mode="eval")
    if not isinstance(node.body, ast.Call):
        raise ValueError("Action不是函数调用。")

    call = node.body
    kwargs: Dict[str, Any] = {}

    # 禁止位置参数、*args、**kwargs
    if call.args:
        raise ValueError("不支持位置参数，只支持关键字参数。")
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError("不支持**kwargs。")
        if not isinstance(kw.value, ast.Constant):
            raise ValueError("参数值只允许常量字符串/数字。")
        kwargs[kw.arg] = kw.value.value

    return func_name, kwargs


def _extract_action(llm_output: str) -> str:
    """
    提取第一条Action行。
    """
    m = re.search(r"^Action:\s*(.+)$", llm_output, flags=re.MULTILINE)
    if not m:
        raise ValueError("模型输出中未找到Action行。")
    return m.group(1).strip()


def main() -> None:
    # 建议用环境变量注入，不要硬编码
    api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "YOUR_BASE_URL")
    model_id = os.environ.get("OPENAI_MODEL", "YOUR_MODEL_ID")

    # TavilyKey也建议走环境变量（search_attraction.py会读取TAVILY_API_KEY）
    # os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"

    llm = OpenAICompatibleClient(model=model_id, api_key=api_key, base_url=base_url)

    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    prompt_history = [f"用户请求:{user_prompt}"]

    print(f"用户输入:{user_prompt}\n" + "=" * 40)

    for i in range(8):
        print(f"--- 循环{i + 1}---\n")

        full_prompt = "\n".join(prompt_history)
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT).strip()

        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        try:
            action_str = _extract_action(llm_output)
        except Exception as e:
            print(f"解析错误:{e}")
            break

        if action_str.startswith("finish"):
            try:
                _, kwargs = _parse_call(action_str)
                final_answer = str(kwargs.get("answer", "")).strip()
            except Exception:
                # 兜底：尽量把finish(...)内部内容吐出来
                final_answer = action_str
            print(f"任务完成，最终答案:{final_answer}")
            break

        try:
            tool_name, kwargs = _parse_call(action_str)
        except Exception as e:
            observation = f"错误:Action解析失败-{e}"
        else:
            tool = available_tools.get(tool_name)
            if not tool:
                observation = f"错误:未定义的工具'{tool_name}'"
            else:
                try:
                    observation = tool(**kwargs)
                except Exception as e:
                    observation = f"错误:工具执行异常-{e}"

        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)


if __name__ == "__main__":
    main()
