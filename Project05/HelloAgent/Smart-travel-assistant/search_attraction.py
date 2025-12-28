import os
from tavily import TavilyClient


def get_attraction(city: str, weather: str) -> str:
    """
    基于城市与天气，调用TavilySearchAPI搜索并返回更“适配天气”的景点推荐。
    """
    api_key = os.environ.get("TAVILY_API_KEY")  # 从环境变量读取API Key
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable not configured."

    tavily = TavilyClient(api_key=api_key)

    # 更自然的检索query（不强行加引号）
    query = f"{city} best tourist attractions suitable for {weather} weather with reasons"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        if response.get("answer"):
            return response["answer"]

        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result.get('title', '')}: {result.get('content', '')}")

        if not formatted_results:
            return "Sorry, no relevant tourist attraction recommendations found."

        return "Based on search, found the following information for you:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"Error: Problem occurred when executing Tavily search - {e}"
