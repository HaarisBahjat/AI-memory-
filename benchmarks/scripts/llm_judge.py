import os
import sys
import json
import asyncio
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from app.services.embedding_service import get_openai_client, settings

class JudgeScore(BaseModel):
    is_correct: bool
    is_supported: bool
    abstention_correct: bool
    reasoning: str

async def evaluate_answer(query: str, ground_truth: str, generated_answer: str, retrieved_context: str, category: str) -> JudgeScore:
    """
    Uses an LLM (gpt-4o-mini) to judge whether the generated answer is correct, 
    supported by the context, and properly handles abstention (NO_ANSWER).
    """
    client = get_openai_client()
    
    prompt = f"""You are an impartial evaluator for a Retrieval-Augmented Generation (RAG) system.
You will be provided with:
1. A User Query
2. The Expected Ground Truth Answer (or NULL if it is unanswerable)
3. The Generated Answer from the RAG system
4. The Retrieved Context used to generate the answer
5. The Query Category ({category})

Evaluate the generated answer based on these criteria:
- is_correct: Does the generated answer match the factual intent of the ground truth? (If ground truth is NULL, is_correct is True if the model abstains).
- is_supported: Is the generated answer fully supported by the retrieved context without hallucination?
- abstention_correct: For NO_ANSWER queries, did the model properly abstain? (For other categories, return True if it attempted an answer).

Respond ONLY with a JSON object in this exact format:
{{
  "is_correct": true/false,
  "is_supported": true/false,
  "abstention_correct": true/false,
  "reasoning": "brief explanation"
}}

Query: {query}
Expected Ground Truth: {ground_truth if ground_truth else "NULL (Should abstain)"}
Retrieved Context: {retrieved_context}

Generated Answer: {generated_answer}
"""

    response = await client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        return JudgeScore(**data)
    except Exception as e:
        print(f"Failed to parse LLM response: {content}")
        return JudgeScore(is_correct=False, is_supported=False, abstention_correct=False, reasoning=str(e))
