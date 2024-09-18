from azure.search.documents.aio import SearchClient
from azure.core.credentials import AzureKeyCredential
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
import asyncio
from tqdm.asyncio import tqdm_asyncio
import aiofiles

load_dotenv()

# read config
AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")
AUZRE_SEARCH_API_KEY=os.getenv("AZURE_COGNITIVE_SEARCH_KEY")

# Define a Markdown template
def format_document_as_markdown(document):
    caption = document.get('caption', 'N/A')
    content = document.get('content', 'N/A')
    image_url = document.get('imageUrl', 'N/A')
    ocr_content = document.get('ocrContent', 'N/A')

    markdown_content = []

    if caption:
        markdown_content.append(f"## {caption}")
    
    if content:
        markdown_content.append(f"#### 主要内容:\n{content}")
    
    if ocr_content:
        markdown_content.append(f"#### 从图片中获取的信息:\n{ocr_content}")
    
    if image_url:
        markdown_content.append(f"\n图片url地址：:\n{image_url}")

    return "\n\n".join(markdown_content)

async def export_all_documents():
    credential = AzureKeyCredential(AUZRE_SEARCH_API_KEY)
    client = SearchClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=credential)
    output_directory = 'search_export'
    os.makedirs(output_directory, exist_ok=True)


    results_per_page = 1000
    skip = 0
    total_count = 0
    page_number = 1
    
    async with client:
        # get total docs
        result = await client.search("*", top=0, include_total_count=True)
        total_count = await result.get_count()

        progress_bar = tqdm_asyncio(total=total_count, desc="Exporting documents")

        while True:
            results = await client.search(
                "*", 
                top=results_per_page, 
                skip=skip, 
                include_total_count=True,
                select=["id", "caption", "content", "imageUrl", "ocrContent"])

            documents = [doc async for doc in results]
            if not documents:
                break 

            # create sub-directory
            page_dir = os.path.join(output_directory, f"page_{page_number}")
            os.makedirs(page_dir, exist_ok=True)

            for document in documents:
                # get docId as filename
                doc_id = document.get('id')
                file_path = os.path.join(page_dir, f"{doc_id}.txt")

                markdown_content = format_document_as_markdown(document)

                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(markdown_content)

                progress_bar.update(1)

            skip += results_per_page
            page_number += 1

        progress_bar.close()
                
    return total_count

async def export_documents(client, results_per_page):
    skip = 0
    while True:
        results = await client.search("*", top=results_per_page, skip=skip, include_total_count=True)
        if not results:
            break
        for result in results:
            yield result
        skip += results_per_page

if __name__ == "__main__":
    allRecordCount = asyncio.run(export_all_documents())
    print("export {} records successfully.",allRecordCount)
