#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''@File : Embedding.py
@Time: 2025/06/20 13:50:47
@Author: Don't have scallions, ginger, garlic
@Version: 1.1
@Desc: None'''

import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        """Initialize the embedded base class
        Args:
            path (str): The path to the model or data
            is_api (bool): Whether to use the API method. True means using online API service, False means using local model"""
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """Get the embed vector representation of text
        Args:
            text (str): Enter text
            model (str): The name of the model used
        Returns:
            List[float]: Embedding vector for text
        Raises:
            NotImplementedError: This method needs to be implemented in a subclass"""
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """Calculate the cosine similarity between two vectors
        Args:
            vector1 (List[float]): First vector
            vector2 (List[float]): The second vector
        Returns:
            float: The cosine similarity between two vectors, ranging from [-1,1]"""
        # Convert input list to numpy array and specify data type float32
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)

        # Check whether the vector contains infinity or NaN values
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0

        # Calculate the dot product of the vector
        dot_product = np.dot(v1, v2)
        # Calculate the norm of the vector (length)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Calculate the denominator (the product of two vector norms)
        magnitude = norm_v1 * norm_v2
        # Handle special cases where the denominator is 0
        if magnitude == 0:
            return 0.0
            
        # Returns cosine similarity
        return dot_product / magnitude
    

class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.client = OpenAI()
            # Get silicon-based flow key from environment variables
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            # Get the basic URL of silicon-based flow from environment variables
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        """Here, the free embedding model with trajectory flow is used by default. BAAI/bge-m3"""
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError
