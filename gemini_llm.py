import os
import json
import requests
from typing import Optional, List, Dict, Union
from dataclasses import dataclass
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

@dataclass
class ProxyConfig:
    host: str = "139.28.233.160"
    port: int = 42567
    user: str = "user3504162"
    password: str = "WGauOMTOpf"

    @property
    def proxy_url(self) -> str:
        return f"http://{self.user}:{self.password}@{self.host}:{self.port}"

class GeminiLLM:
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.95,
        max_output_tokens: int = 8192,
        proxy_config: Optional[ProxyConfig] = None
    ):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        self.proxy_config = proxy_config or ProxyConfig()
        self.json_mode = False
        
        self.generation_config = {
            "temperature": temperature,
            "topK": top_k,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "text/plain"
        }

    def _prepare_request_data(self, prompt: str) -> Dict:
        """Подготовка данных для запроса"""
        return {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": self.generation_config
        }

    def _make_request(self, data: Dict) -> Dict:
        """Выполнение HTTP запроса к API"""
        headers = {"Content-Type": "application/json"}
        proxies = {"http": self.proxy_config.proxy_url, "https": self.proxy_config.proxy_url}
        
        response = requests.post(
            f"{self.base_url}?key={self.api_key}",
            headers=headers,
            json=data,
            proxies=proxies
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
            
        return response.json()

    def generate_response(self, prompt: str, chat_history: Optional[List[Dict]] = None) -> Dict:
        """Генерация ответа на промпт с учетом истории чата
        
        Args:
            prompt (str): Текст промпта
            chat_history (Optional[List[Dict]]): История чата в формате списка словарей 
                [{"role": "user/model", "parts": [{"text": "сообщение"}]}]
                
        Returns:
            Dict: Словарь с ответом и информацией о токенах
        """
        try:
            # Подготавливаем базовый запрос с текущим промптом
            data = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": self.generation_config
            }
            
            # Если есть история чата, добавляем её перед текущим промптом
            if chat_history:
                data["contents"] = [
                    {
                        "role": msg["role"],
                        "parts": msg["parts"]
                    }
                    for msg in chat_history
                ] + data["contents"]
                
            # Делаем запрос к API
            response = self._make_request(data)
            
            # Извлекаем ответ и информацию о токенах
            model_response = response["candidates"][0]["content"]["parts"][0]["text"]
            token_info = response.get("usageMetadata", {})
            
            # Возвращаем структурированный ответ
            return {
                "text": model_response,
                "tokens": {
                    "prompt": token_info.get("promptTokenCount", 0),
                    "completion": token_info.get("candidatesTokenCount", 0),
                    "total": token_info.get("totalTokenCount", 0)
                }
            }
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "text": f"Error: {str(e)}",
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }

    def set_json_output(self):
        """Переключение на JSON формат вывода"""
        self.generation_config["responseMimeType"] = "application/json"
        self.json_mode = True

    def set_text_output(self):
        """Переключение на текстовый формат вывода"""
        self.generation_config["responseMimeType"] = "text/plain"
        self.json_mode = False

if __name__ == "__main__":
    # Пример использования
    llm = GeminiLLM()
    
    # Получение полного ответа API
    print("\nПолный ответ API:")
    response = llm.generate_response("Расскажи о квантовой физике простыми словами")
    print(json.dumps(response, ensure_ascii=False, indent=2))
    
    # JSON формат
    print("\nJSON формат:")
    llm.set_json_output()
    response = llm.generate_response("Назови 3 основных принципа квантовой механики")
    print(json.dumps(response, ensure_ascii=False, indent=2))