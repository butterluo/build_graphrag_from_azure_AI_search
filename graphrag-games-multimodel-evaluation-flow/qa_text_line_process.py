
from dataClass import LineEvaluationResult
from promptflow.core import tool
import requests
import json
from promptflow.core import Prompty
from fastapi import FastAPI, HTTPException


import re


# regular parttern URL
url_pattern = r'https?://[^\s\[\],]+'

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def qa_text_line_process(question: str,gameName:str,imageUrl:list[str],answer:str) -> dict[str:float]:

    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # 使用正则表达式提取所有的 URL
    imageUrls = re.findall(url_pattern, imageUrl)


    url = "http://localhost:8000/v1/multimode/chat"
    headers = {
    "Content-Type": "application/json"}
    data = {
        "prompt": "在游戏: " + gameName + "中，" + question,
        "imageUrl": imageUrls[0]
    }

    response = requests.post(url, json=data, headers=headers)
    lineEvaluationResult:LineEvaluationResult = None

    if response.status_code == 200:
        print("Response:", response.json())
        lineEvaluationResult = json.loads(response.text)
    else:
        print("Error:", response.status_code)
        raise HTTPException(status_code=response.status_code, detail=response.reason) 

    retrieve_context_relevance_prompty = Prompty.load(source="/home/azureuser/build_graphrag_from_azure_AI_search/graphrag-games-evaluation-flow/prompty/retrieve_context_relevance.prompty")
    groundedness_prompty = Prompty.load(source="/home/azureuser/build_graphrag_from_azure_AI_search/graphrag-games-evaluation-flow/prompty/groundedness.prompty")
    relevance_prompty = Prompty.load(source="/home/azureuser/build_graphrag_from_azure_AI_search/graphrag-games-evaluation-flow/prompty/relevance.prompty")

    retrieve_context_relevance_score = retrieve_context_relevance_prompty(context=lineEvaluationResult["context_text"],question=question)
    groundedness_score = groundedness_prompty(answer=lineEvaluationResult["response"],context=answer)
    relevance_score = relevance_prompty(question=question,answer=lineEvaluationResult["response"],context=lineEvaluationResult["context_text"])


    
    return {"retrieve_context_relevance_score":retrieve_context_relevance_score,"groundedness_score":groundedness_score,"relevance_score":relevance_score}