from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

class Vec():
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        self.openai_embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
    
    def extractData(self):
        try:
            loader = TextLoader('대구공고전문.txt', encoding='utf-8')
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators = '\n')
            texts = text_splitter.split_documents(pages)

            shutil.rmtree('vector_db', ignore_errors=True)
            self.vector_db = Chroma.from_documents(
                documents = texts,
                embedding = self.openai_embedding,
                persist_directory='vector_db'
            )
        except Exception as e:
            return e
        
vec = Vec()
vec.extractData()