#from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


# could use this OLlama embeddings for local LLM 
#from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
# def get_embedding_function():
#     # llama3: https://ollama.com/library/llama3
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     return embeddings


# def get_embedding_function():
#     #embeddings = OpenAIEmbeddings()
#     embeddings = OllamaEmbeddings(model="llama3:8b-instruct-q4_0")
#     return embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

def get_embedding_function():
    # This code snippet is checking if a CUDA-enabled GPU is available for use. If a CUDA-enabled GPU
    # is available, the variable `device` is set to "cuda", indicating that the GPU will be used for
    # computations. If a CUDA-enabled GPU is not available, the variable `device` is set to "cpu",
    # indicating that the computations will be done on the CPU instead. The `print` statement is then
    # used to display which device (GPU or CPU) will be used for processing.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"Using device: {device}")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    #embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large", model_kwargs={'device': device})
    return embeddings