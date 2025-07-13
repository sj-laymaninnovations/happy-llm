from src.core import Agent
from src.tools import add, count_letter_in_string, compare, get_current_datetime, search_wikipedia, get_current_temperature

from openai import OpenAI


if __name__ == "__main__":
    client = OpenAI(
        api_key="your siliconflow api key",
        base_url="https://api.siliconflow.cn/v1",
    )

    agent = Agent(
        client=client,
        model="Qwen/Qwen2.5-32B-Instruct",
        tools=[get_current_datetime, search_wikipedia, get_current_temperature],
    )

    while True:
        # Use color output to distinguish user input from AI answers
        prompt = input("\033[94mUser: \033[0m")  # Blue display user input prompt
        if prompt == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # Green display AI assistant answer
