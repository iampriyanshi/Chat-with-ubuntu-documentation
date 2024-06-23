## TODO Add imports
import json
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.llms import HuggingFacePipeline, Ollama
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch



# from app import logger
## TODO Add below:
# 1. errors/exception handling 
# 2. loggers

def initialize_vectorstore(vectorstore_path, embeddings, logger):
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=vectorstore_path,
        collection_name="ubuntu_docs",
    )
    logger.info(f"Vectorstore initialization completed")
    return vectorstore

def get_templates(path):
    with open(f"{path}/sample.txt", "r") as f:
        template_sample = f.read()
    return {"sample_prompt_template":template_sample}

def get_top_documents(response):
    pass

def init_config():
    with open('demo_bot/data/metadata/config.json') as fobj:
        config_data = json.load(fobj)
    return config_data

def get_title_from_content(doc):
    print(doc)
    title_key = "title: "
    title = ''
    start = doc.find(title_key)
    if start != -1:
        end = doc.find("\n", start)
        title = doc[start + len(title_key):end]
        print('title: ', title)
        title = title.replace('"', '')
    return title

def llm_pipeline():
    local_llm = Ollama(
            model="phi3",
            base_url="http://localhost:11434",
            verbose=True,
            # num_gpu=0,
            temperature=0.2,
        )
    return local_llm