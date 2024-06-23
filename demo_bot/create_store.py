from __future__ import print_function
import openai
# from langchain.embeddings import 
# from langchain_community.embeddings import HuggingFaceEmbeddings

import logging
from utils.Utilities import init_config
from datetime import date
from operator import itemgetter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from constants import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from utils.Utilities import get_title_from_content
import glob
import os
import warnings
warnings.filterwarnings("ignore")

##### init logging
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(f'demo_bot/data/logs/Demo_bot_events_{date.today().strftime("%d-%m-%y")}.log')
fh.setFormatter(formatter)
logger.addHandler(fh)
####

def main():
    print('Initializing Vector store ...\n')
    config_data = init_config()
    vectorstore_path, prompt_template_path = itemgetter('vectorstore_path', 'prompt_template_path')(config_data)
    # embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key) 
    directory = "demo_bot_data/ubuntu-docs"

    # Get a list of all .md files in the demo_bot_data/ubuntu-docs directory
    md_files = glob.glob(f'{directory}/**/*.md', recursive=True)

    # Iterate over the list of files
    docs = []
    for filename in md_files:
        # Open each file
        with open(filename, 'r') as f:
            loader = UnstructuredMarkdownLoader(filename)
            docs.append(loader.load())
    print(docs[0])
    #     # Read the file and print its contents
    # loader = DirectoryLoader(directory, glob="**/*.md", use_multithreading=True, show_progress=True, loader_cls=UnstructuredMarkdownLoader)
    # docs = loader.load()
    logger.info(f'Loading documents completed.')

    page_content = []
    metadata = []
    for doc in docs:
        page_content.append(doc.page_content)
        doc.metadata['title'] = get_title_from_content(doc.page_content)
        metadata.append(doc.metadata)

    splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True
        )
    ubuntun_markdown_docs = splitter.create_documents(page_content, metadata)
    for idx, ubuntun_markdown_doc in enumerate(ubuntun_markdown_docs):
        ubuntun_markdown_doc.metadata["id"] = idx
    logger.info(f'Documents split completed.')

    logger.info(f'Setting up the vector store.')
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=ubuntun_markdown_docs,
        embedding=embeddings,
        persist_directory=vectorstore_path,
        collection_name="ubuntu_docs",
    )
    # persisting the db to the disk
    vectorstore.persist()
    vectorstore = None
    logger.info(f'Vector store initialization completed.')

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        logger.error(f"Error encountered in function: [main]: {str(err)}")
