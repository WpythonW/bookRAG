from typing import List, Dict
from pydantic import BaseModel
import time
from tqdm import tqdm
from datetime import datetime
import json

class Scene(BaseModel):
    description: str
    who: List[str]
    location: str
    key_details: List[str]
    chunk_index: int
    scene_index: int

class TokenInfo(BaseModel):
    prompt: int
    completion: int
    total: int
    timestamp: str

class ChunkSummary(BaseModel):
    scenes: List[Scene]
    chunk_index: int
    token_info: TokenInfo

class ProcessingStats(BaseModel):
    start_time: str
    end_time: str
    total_chunks: int
    errors: int
    successful_chunks: int

class ProcessingResult(BaseModel):
    chunks: List[ChunkSummary]
    all_scenes: List[Scene]
    total_tokens: Dict[str, int]
    processing_stats: ProcessingStats

class TextProcessor:
    def __init__(self, gemini_llm):
        self.llm = gemini_llm
        self.request_count = 0
        
    def get_previous_context(self, previous_summaries: List[ChunkSummary], n_previous: int = 3) -> str:
        if not previous_summaries:
            return ""
            
        start_idx = max(0, len(previous_summaries) - n_previous)
        recent_summaries = previous_summaries[start_idx:]
        
        context = {
            "key_events": [],
            "active_characters": set(),
            "locations": set()
        }
        
        for summary in recent_summaries:
            for scene in summary.scenes:
                context["key_events"].append(scene.description)
                context["active_characters"].update(scene.who)
                context["locations"].add(scene.location)
        
        return f"""Previous context:
Key events: {' | '.join(context['key_events'][-3:])}
Active characters: {', '.join(context['active_characters'])}
Recent locations: {', '.join(context['locations'])}"""

    def get_chunk_summary(
        self,
        current_chunk: str,
        chunk_index: int,
        previous_summaries: List[ChunkSummary],
        n_previous_summaries: int = 3
    ) -> tuple[ChunkSummary, TokenInfo]:
        # Проверяем счетчик запросов
        self.request_count += 1
        if self.request_count % 10 == 0:
            print("\nПауза на 30 секунд...")
            time.sleep(30)
        
        previous_context = self.get_previous_context(previous_summaries, n_previous_summaries)
        
        prompt = f'''Summarize the text chunk as JSON. Previous context for reference only.

{previous_context}

START CHUNK:
{current_chunk}
END CHUNK

Format:
{{
  "scenes": [
    {{
      "description": "1-2 sentences describing key events",
      "who": ["characters present"],
      "location": "scene location",
      "key_details": ["1-2 important facts"]
    }}
  ]
}}

Requirements:
- Split into multiple scenes if location/time changes
- Keep descriptions brief but clear
- Include main characters and key events
- No empty fields allowed'''

        empty_token_info = TokenInfo(
            prompt=0,
            completion=0,
            total=0,
            timestamp=datetime.now().isoformat()
        )

        self.llm.set_json_output()
        try:
            response = self.llm.generate_response(prompt)
            summary_data = response["text"]
            tokens_data = response.get("tokens", {"prompt": 0, "completion": 0, "total": 0})
            
            current_token_info = TokenInfo(
                prompt=tokens_data['prompt'],
                completion=tokens_data['completion'],
                total=tokens_data['total'],
                timestamp=datetime.now().isoformat()
            )
            
            raw_data = json.loads(summary_data)
            
            for i, scene in enumerate(raw_data.get('scenes', [])):
                scene['chunk_index'] = chunk_index
                scene['scene_index'] = i
            
            enhanced_data = {
                "scenes": raw_data.get('scenes', []),
                "chunk_index": chunk_index,
                "token_info": {
                    "prompt": current_token_info.prompt,
                    "completion": current_token_info.completion,
                    "total": current_token_info.total,
                    "timestamp": current_token_info.timestamp
                }
            }
            
            summary_obj = ChunkSummary.model_validate(enhanced_data)
            
            self.llm.set_text_output()
            return summary_obj, current_token_info
            
        except Exception as e:
            print(f"Error in chunk {chunk_index}: {str(e)}")
            self.llm.set_text_output()
            
            empty_summary = ChunkSummary(
                scenes=[],
                chunk_index=chunk_index,
                token_info=empty_token_info
            )
            return empty_summary, empty_token_info
            
    def process_book(self, chunks: List[str]) -> ProcessingResult:
        summaries = []
        all_scenes = []
        total_tokens = {"prompt": 0, "completion": 0, "total": 0}
        processing_stats = {
            "start_time": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "errors": 0,
            "successful_chunks": 0
        }
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing text")):
            try:
                summary, token_info = self.get_chunk_summary(
                    current_chunk=chunk,
                    chunk_index=i,
                    previous_summaries=summaries
                )
                
                if token_info:
                    total_tokens["prompt"] += token_info.prompt
                    total_tokens["completion"] += token_info.completion
                    total_tokens["total"] += token_info.total
                    print(f"\nTokens for chunk {i}: Prompt: {token_info.prompt}, "
                          f"Completion: {token_info.completion}, Total: {token_info.total}")
                
                summaries.append(summary)
                all_scenes.extend(summary.scenes)
                processing_stats["successful_chunks"] += 1
                time.sleep(1)
                
            except Exception as e:
                print(f"\nError in chunk {i}: {str(e)}")
                processing_stats["errors"] += 1
                print("Retrying in 5 seconds...")
                time.sleep(5)
                try:
                    summary, token_info = self.get_chunk_summary(
                        current_chunk=chunk,
                        chunk_index=i,
                        previous_summaries=summaries
                    )
                    summaries.append(summary)
                    all_scenes.extend(summary.scenes)
                    processing_stats["successful_chunks"] += 1
                except Exception as e:
                    print(f"Retry failed: {str(e)}")
                    processing_stats["errors"] += 1
                    empty_token_info = TokenInfo(
                        prompt=0,
                        completion=0,
                        total=0,
                        timestamp=datetime.now().isoformat()
                    )
                    empty_summary = ChunkSummary(
                        scenes=[],
                        chunk_index=i,
                        token_info=empty_token_info
                    )
                    summaries.append(empty_summary)
        
        processing_stats["end_time"] = datetime.now().isoformat()
        
        return ProcessingResult(
            chunks=summaries,
            all_scenes=all_scenes,
            total_tokens=total_tokens,
            processing_stats=ProcessingStats(
                start_time=processing_stats["start_time"],
                end_time=processing_stats["end_time"],
                total_chunks=processing_stats["total_chunks"],
                errors=processing_stats["errors"],
                successful_chunks=processing_stats["successful_chunks"]
            )
        )