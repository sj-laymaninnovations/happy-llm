#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''@File : utils.py
@Time: 2025/06/20 13:50:47
@Author: Don't have scallions, ginger, garlic
@Version: 1.1
@Desc: None'''

import os
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import markdown
import json
from tqdm import tqdm
import tiktoken
from bs4 import BeautifulSoup
import re

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        # args: dir_path, destination folder path
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # The os.walk function will recursively traverse the specified folder
            for filename in filenames:
                # Determine whether the file type meets the requirements by suffix name
                if filename.endswith(".md"):
                    # If the requirements are met, add its absolute path to the result list
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        # Read file content
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []

        curr_len = 0
        curr_chunk = ''

        token_len = max_token_len - cover_content
        lines = text.splitlines()  # Assume that text is divided into lines with newline characters

        for line in lines:
            # Keep spaces, remove only the first and last spaces of the line
            line = line.strip()
            line_len = len(enc.encode(line))
            
            if line_len > max_token_len:
                # If the length of a single line exceeds the limit, divide it into multiple blocks
                # Save the current block first (if there is content)
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                    curr_chunk = ''
                    curr_len = 0
                
                # Split long lines by token length
                line_tokens = enc.encode(line)
                num_chunks = (len(line_tokens) + token_len - 1) // token_len
                
                for i in range(num_chunks):
                    start_token = i * token_len
                    end_token = min(start_token + token_len, len(line_tokens))
                    
                    # Decode token fragments to text
                    chunk_tokens = line_tokens[start_token:end_token]
                    chunk_part = enc.decode(chunk_tokens)
                    
                    # Add overlay content (except the first block)
                    if i > 0 and chunk_text:
                        prev_chunk = chunk_text[-1]
                        cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                        chunk_part = cover_part + chunk_part
                    
                    chunk_text.append(chunk_part)
                
                # Reset the current block status
                curr_chunk = ''
                curr_len = 0
                
            elif curr_len + line_len + 1 <= token_len:  # +1 for newline
                # The current line can be added to the current block
                if curr_chunk:
                    curr_chunk += '\n'
                    curr_len += 1
                curr_chunk += line
                curr_len += line_len
            else:
                # The current line cannot be added to the current block, start a new block
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                
                # Start a new block and add overlay content
                if chunk_text:
                    prev_chunk = chunk_text[-1]
                    cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                    curr_chunk = cover_part + '\n' + line
                    curr_len = len(enc.encode(cover_part)) + 1 + line_len
                else:
                    curr_chunk = line
                    curr_len = line_len

        # Add the last block (if there is content)
        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str):
        # Select the reading method according to the file extension
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        # Read PDF files
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path: str):
        # Read Markdown file
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # Extract plain text from HTML using BeautifulSoup
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # Use regular expression to remove URL link
            text = re.sub(r'http\S+', '', plain_text) 
            return text

    @classmethod
    def read_text(cls, file_path: str):
        # Read text files
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()


class Documents:
    """Get the json format document that has been classified"""
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content
