from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import shutil
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

documents = []
loader = TextLoader('대구공고전문.txt', encoding='utf-8')
content = loader.load()[0].page_content
texts = content.split('\n\n\n')
for i in range(len(texts)):
    doc =  Document(page_content=texts[i], metadata={"source": "대구공고100년사.txt"})
    documents.append(doc)

class Vec():
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        self.openai_embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
    
    def extractData(self):
        try:
            shutil.rmtree('vector_db', ignore_errors=True)
            self.vector_db = Chroma.from_documents(
                documents = documents,
                embedding = self.openai_embedding,
                persist_directory='vector_db'
            )
        except Exception as e:
            return e

"""#내용 업데이트 시 실행       
vec = Vec()
errmsg = vec.extractData()
if errmsg:
    print(errmsg)
else:
    print(f"vector_db 업데이트 완료. ({datetime.now()})")"""