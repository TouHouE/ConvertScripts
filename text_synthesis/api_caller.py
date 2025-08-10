import json
import os
import time
from os.path import join, exists
from typing import ClassVar, Any

import google.generativeai as genai
from google.generativeai.types import generation_types

from google import genai as genai2
from google.genai import types

def load_file(path) -> str:
    with open(path, 'r', encoding='utf-8') as reader:
        return reader.read()


def load_json(path):
    with open(path, 'r') as jin:
        return json.load(jin)


class GeminiCaller:    
    User_Template: ClassVar[str]
    
    @classmethod
    def set_User_Template(cls, User_Template: str):
        cls.User_Template = User_Template

    def __init__(self, args):
        self.token_map: dict[str, bool] = load_json(args.token_path)
        self.system_message = load_file(args.system_message_file)
        self.update_model()
    
    def reset_api_token(self):
        """Change Faild, try to check user will given new token or not."""
        if exists(self.args.newer_token_json):
            num_new_token = 0
            for new_token in load_json(self.args.newer_token_json):
                if new_token in self.token_map.keys():
                    continue
                num_new_token += 1
                self.token_map[new_token] = False
            if num_new_token > 0:
                return self.update_model()
            
        """No new token given, try to reloop all exhaust token after sleeping `args.sleep`"""        
        print(f'Because all of Token are exhausted, we will rest some time...')
        time.sleep(self.args.sleep)
        print(f'Ok, Let we continue')
        for token in self.token_map:
            self.token_map[token] = False
        return self.update_model()
    
    def update_model(self):
        """Try to change workable token"""
        is_updated = False
        for token, is_exhaust in self.token_map.items():
            if is_exhaust:
                continue
            genai.configure(api_key=token)
            self.model = genai.GenerativeModel(
                self.args.gemini_model_name, system_instruction=self.system_message
            )
            is_updated = True
        if is_updated:
            return
        self.reset_api_token()
        

    def __call__(self, content_list: Any | list[Any]) -> types.GenerateContentResponse:
        assert GeminiCaller.User_Template is not None, 'User Template cannot be None, please setting User_Template'
        if not isinstance(content_list, list):
            content_list: list[str] = [content_list]
        msg = GeminiCaller.User_Template.format(*content_list)

        while True:
            try:
                response = self.model.generate_content(msg)
                return response
            except Exception as e:
                self.update_model


class GeminiCallerV2(GeminiCaller):
    def __init__(self, args):
        self.token_map: dict[str, bool] = load_json(args.token_path)
        self.system_message = load_file(args.system_message_file)
        self.args = args        
        self.model_name = args.gemini_model_name
        self.update_model()
    
    def update_model(self):
        """Try to change workable token"""
        is_updated = False
        for token, is_exhaust in self.token_map.items():
            if is_exhaust:
                continue
            self.client = genai.Client(api_key=token)
            is_updated = True
        if is_updated:
            return
        self.reset_api_token()
    
    def __call__(
            self, old_query: str, old_answer: str,
            temp: float = 0.7, top_p: float = 0.9, thinking_mode: bool = False
    ) -> generation_types.GenerateContentResponse:
        
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=GeminiCallerV2.User_Template.replace(
                        '[The medical question]', old_query
                    ).replace(
                        '[The answer currently in the dataset]', old_answer
                    ),
                    config=types.GenerateContentConfig(
                        temperature=temp,
                        top_p=top_p,
                        system_instruction=self.system_message,
                        thinking_config=types.ThinkingConfig(thinking_budget=0) if not thinking_mode else None
                    ),
                )
                return response
            except Exception as e:
                print(f'Error: {e}')
                self.update_model()
                