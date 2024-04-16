#%%
import os
import sys
import json
import logging
import requests
import pandas as pd
from openai import AzureOpenAI, OpenAI
#%%
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
file_handler = logging.FileHandler('modellog.txt', mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
#%%
class FFMLLama2:
    def __init__(self, switch_zh_hans=False) -> None:
        self.api_key="af13849c-85d7-4ccf-8879-89c58052ea78"
        self.model_name="ffm-llama2-70b-chat-exp"
        self.api_url="https://57729.afs.twcc.ai/text-generation"
        self.max_new_tokens=2048
        self.temperature=0.5
        self.top_k=50
        self.top_p=1.0
        self.frequency_penalty=1.0

    def conversation(self, text, system):
        headers = {
            "content-type": "application/json",
            "X-API-Key": self.api_key}
        
        data = {
                "model":self.model_name,
                "message": [
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "human",
                        "content": text
                    }
                ],
                "parameters": {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty
                }
            }
        res = ""
        try:
            response = requests.post(self.api_url + "/api/models/conversation", json=data, headers=headers, verify=False)
            res = json.loads(response.text, strict=False)['generated_text']
            logger.info(f'{res}')
            with open('answer_record.json', 'a', encoding='utf-8-sig') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
        except Exception as er:
            print(response.status_code)
        if res != None:
            res = res.strip("\n")
        else:
            res = ""
        return res
#%%
class AzureGPT35:
    """
    key1
    33579d1535d046d4a3ae2655027f2445

    key2
    6a5103d27e3d44f19342e52a79426e17

    Endpoint
    https://azheopenai.openai.azure.com/

    model_name
    gpt-35-turbo-16k
    gpt-35-turbo-instruct
    gpt-4-32k
    text-embedding-ada-002
    """
    def __init__(self) -> None:
        self.client = AzureOpenAI(
            api_key="31db5c7e68a4454ba8e3082918804ea2",
            api_version="2023-05-15",
            azure_endpoint="https://skfh-aoai.openai.azure.com/",
            default_headers= {"conten-type": "application/json"}
        )
        self.model="gpt-35-turbo-16k"
        self.temperature = 0.5
        self.top_p = 1.0
        self.frequency_penalty = 1.0

    def conversation(self, system, text):

        message = {
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        }

        try:
            response = self.client.chat.completions.create(
                messages= message,
                model= self.model,
                max_tokens=2500,
                frequency_penalty= self.frequency_penalty,
                temperature= self.temperature,
                top_p= self.top_p
            )
            logger.info('-'*50)
            logger.info(f'{response}')
        except Exception as msg:
            print(msg)

        return response.choices[0].message.content
#%%
class BreeXe8x7b:
    def __init__(self) -> None:
        self.model = "BreeXe-8x7B"
        self.temperature = 0.5
        self.top_p = 1.0
        self.max_tokens=512

    def conversation(self, system, text):

        client = OpenAI(
            base_url= "https://api-mtkresearch.com/v1/",
            api_key=""
        )

        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ]

        try:
            response = client.chat.completions.create(
                messages= message,
                model= self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            logger.info('-'*50)
            logger.info(f'{response}')
        except Exception as msg:
            print(msg)
        
        return response.choices[0].message.content
#%%
class Breeze7b:
    def __init__(self) -> None:
        self.model = "Repository/breeze7gguf/breeze-7b-instruct-v1_0-q6_k.gguf"
        self.temperature = 0.5
        self.top_p = 1.0
        self.max_tokens=512

    def conversation(self, system, text):

        client = OpenAI(
            base_url= "https://localhost:1234/v1",
            api_key=""
        )

        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ]
        try:
            response = client.chat.completions.create(
                messages= message,
                model= self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            logger.info('-'*50)
            logger.info(f'{response}')
        except Exception as msg:
            print(msg)

        return response.choices[0].message.content
#%%