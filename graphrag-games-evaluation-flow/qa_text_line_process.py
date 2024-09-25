
from dataClass import LineEvaluationResult
from promptflow.core import tool
import requests
import json
from promptflow.core import Prompty

# settings.py
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
load_dotenv(verbose=True)
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path, verbose=True)

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def qa_text_line_process(question: str,answer:str) -> dict[str:float]:

    import sys,os
    # print(sys.path)
    print(os.environ) 
    print("-*-"*10)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    url = "http://localhost:8012/v1/chat/completions"
    headers = {
    "Content-Type": "application/json"}
    data = {
        "prompt": question
    }

    response = requests.post(url, json=data, headers=headers)
    lineEvaluationResult:LineEvaluationResult = None

    if response.status_code == 200:
        print("Response:", response.json())
        lineEvaluationResult = json.loads(response.text)
    else:
        print("Error:", response.status_code)

    retrieve_context_relevance_prompty = Prompty.load(source="/home/luogang/SRC/NLP/LLM/build_graphrag_from_azure_AI_search/graphrag-games-evaluation-flow/prompty/retrieve_context_relevance.prompty")
    groundedness_prompty = Prompty.load(source="/home/luogang/SRC/NLP/LLM/build_graphrag_from_azure_AI_search/graphrag-games-evaluation-flow/prompty/groundedness.prompty")
    relevance_prompty = Prompty.load(source="/home/luogang/SRC/NLP/LLM/build_graphrag_from_azure_AI_search/graphrag-games-evaluation-flow/prompty/relevance.prompty")

    retrieve_context_relevance_score = retrieve_context_relevance_prompty(context=lineEvaluationResult["context_text"],question=question)
    groundedness_score = groundedness_prompty(answer=lineEvaluationResult["response"],context=answer)
    relevance_score = relevance_prompty(question=question,answer=lineEvaluationResult["response"],context=lineEvaluationResult["context_text"])


    
    return {"retrieve_context_relevance_score":retrieve_context_relevance_score,"groundedness_score":groundedness_score,"relevance_score":relevance_score}