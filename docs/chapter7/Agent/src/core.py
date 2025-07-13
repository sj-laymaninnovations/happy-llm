from openai import OpenAI
import json
from typing import List, Dict, Any
from src.utils import function_to_json
from src.tools import get_current_datetime, add, compare, count_letter_in_string, search_wikipedia, get_current_temperature

import pprint

SYSTEM_PROMPT = """You are an artificial intelligence assistant called "Don't have onions, ginger and garlic". Your output should be consistent with the user's language.
When a user's problem requires calling a tool, you can call the appropriate tool function from the provided tool list."""

class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools: List=[], verbose : bool = True):
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # Get the JSON pattern for all tools
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # Processing tool calls
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        function_call_content = eval(f"{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        self.messages.append({"role": "user", "content": prompt})

        # Get the completion response of the model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        if response.choices[0].message.tool_calls:
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            # Processing tool calls
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                # Process tool calls and add results to the message list
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print("Calling tools:", response.choices[0].message.content, tool_list)
            # Get the completion response of the model again, this time including the result of the tool call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # Add the model's completion response to the message list
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content


    

