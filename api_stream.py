from unstruct_step1 import Vec
import asyncio
from typing import AsyncIterable
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote.retrievers import KkmaBM25Retriever
from langchain.retrievers import EnsembleRetriever
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from konlpy.tag import Kkma
from datetime import datetime
import os

MAX_CHAT_HISTORY = 5

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name = "static")

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

if not os.path.exists('vector_db'):
    vec = Vec()
    errmsg = vec.extractData()
    if errmsg:
        print(errmsg)
    else:
        print(f"vector_db 업데이트 완료. ({datetime.now()})")

openai_embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
persist_directory = 'vector_db'
vector_db = Chroma(
    persist_directory = persist_directory,
    embedding_function = openai_embedding,
)

loader = TextLoader('대구공고전문.txt', encoding='utf-8')
content = loader.load()[0].page_content
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200, separators = '\n')
splitted_docs = text_splitter.split_text(content)

kbm25_retriever = KkmaBM25Retriever.from_texts(splitted_docs, k=3)
embed_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
ensemble_retriever = EnsembleRetriever(retrievers=[kbm25_retriever, embed_retriever], weights=[0.4, 0.6])

class Message(BaseModel):
    content: str

class NoDocumentsRetrievedError(Exception):
    pass

chat_history = []
async def send_message(content: str) -> AsyncIterable[str]:
    try:
        callback = AsyncIteratorCallbackHandler()
        
        kkma = Kkma()
        tok_query = kkma.nouns(content)
        
        docs = ensemble_retriever.invoke(' '.join(tok_query))
        new_docs = list(doc.page_content for doc in docs)
        
        if not new_docs:
          raise NoDocumentsRetrievedError("No documents retrieved.")
        
        model = ChatOpenAI(
            streaming=True,
            verbose=True,
            callbacks=[callback],
        )

        year = datetime.now().year
        history = "\n".join(f"Q: {item['question']}\nA: {item['answer']}" for item in chat_history[-2:])
        print(history)
        template = '''
        너는 대구공업고등학교 80년사 책의 내용과 관련된 질문에 답변하는 챗봇이야. 답변은 한국어 높임말로 써. 일관되고 친절한 말투를 써.
        주어진 데이터와 관련된 질문에만 답변해줘. 올해는 {year}년이야.
        {context}
        Chat history: {history}
        Question: {question}
        Answer:
        '''

        prompt = PromptTemplate(
                    input_variables=[
                        "context",
                        "year",
                        "question",
                        "history"
                    ],
                    template=template
                )
        
        chain = prompt | model | StrOutputParser()
        
        task = asyncio.create_task(
            chain.ainvoke({"year": year, "context": new_docs, "question": content, "history": history})
        )

        async for token in callback.aiter():
            yield token

        response = await task
        chat_history.append({"question": content, "answer": response})
        if len(chat_history) > MAX_CHAT_HISTORY:
            chat_history.pop(0)
            
    except Exception as e:
        yield "죄송합니다. 지금은 답변해 드릴 수 없습니다."
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

@app.post("/stream_chat/")
async def stream_chat(message: Message):
    generator = send_message(message.content)
    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})