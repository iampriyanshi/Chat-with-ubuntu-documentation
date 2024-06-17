## TODO Add imports
import json
## TODO Add below:
# 1. errors/exception handling 
# 2. loggers

def initialize_vectorstore(vectorstore_path, embeddings):
    pass

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
