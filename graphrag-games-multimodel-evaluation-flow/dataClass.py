from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field


@dataclass
class Record:
    sls_id: Optional[str] = None
    game_name: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    images: Optional[List[str]] = None

@dataclass
class LineEvaluationResult:
    completion_time:Optional[str] = None
    context_text:Optional[str] = None
    llm_calls:Optional[str] = None
    prompt_tokens:Optional[str] = None
    response:Optional[str] = None
    accuracy: Optional[float] = None
    rag_accuracy: Optional[float] = None
    
@dataclass
class AggregateResult:
    accuracy: Optional[float] = None
    aggregated_result: Optional[float] = None

class PromptRequest(BaseModel):
    prompt: str