#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''@File : LLM.py
@Time: 2025/06/20 13:50:47
@Author: Don't have scallions, ginger, garlic
@Version: 1.1
@Desc: None'''
import os
from typing import Dict, List, Optional, Tuple, Union
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

RAG_PROMPT_TEMPLATE="""Use context to answer user questions. If you don't know the answer, just say you don't know. Always answer in Chinese.
Question: {question}
Referenced context:
···
{context}
···
If the given context cannot get you to answer, please answer that there is no content in the database, you don't know.
Useful answers:"""


class BaseModel:
    def __init__(self, model) -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass

class OpenAIChat(BaseModel):
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")   
        client.base_url = os.getenv("OPENAI_BASE_URL")
        history.append({'role': 'user', 'content': RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=2048,
            temperature=0.1
        )
        return response.choices[0].message.content
