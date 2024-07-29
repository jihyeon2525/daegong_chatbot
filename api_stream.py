from unstruct_step1 import Vec, documents

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from typing import AsyncIterable, Dict
from pydantic import BaseModel
import asyncio

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from kiwipiepy import Kiwi

from datetime import datetime
from dotenv import load_dotenv
import os

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

kiwi = Kiwi()
def analyze_text(text):
    nouns = []
    key_nouns = []
    particles = ['은', '는', '에', '이', '가', '의']
    result = kiwi.analyze(text)
    for token, pos, _, _ in result[0][0]:
        if len(token) != 1 and pos in ['NNG', 'NNP', 'NR', 'XR', 'SL', 'SH', 'SN']:
            nouns.append(token)
        elif pos in ['JKS', 'JKB', 'JX', 'JKG'] and (token in particles) and nouns:
            key_nouns.append(nouns[-1])
    return nouns, key_nouns

kbm25_retriever = KiwiBM25Retriever.from_documents(documents, k=3)
embed_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
ensemble_retriever = EnsembleRetriever(retrievers=[kbm25_retriever, embed_retriever], weights=[0.5, 0.5])

global_chat_history = {'question': '', 'docs': ''}

class Message(BaseModel):
    content: str
    chat_history: Dict[str, str]

class NoDocumentsRetrievedError(Exception):
    pass

async def send_message(content: str, chat_history: Dict[str, str]) -> AsyncIterable[str]:
    try:
        callback = AsyncIteratorCallbackHandler()

        tok_query, key_nouns = analyze_text(content)
        if not key_nouns:
            key_nouns = tok_query

        if chat_history.get('question'):
            if tok_query:
                question = ' '.join(tok_query)
                docs = ensemble_retriever.invoke(question)
                new_docs = list(set(doc.page_content.replace('\t', ' ') for doc in docs))
                if not new_docs:
                    raise NoDocumentsRetrievedError("No documents retrieved.")
                filtered_docs = " ".join([f"<Doc{i+1}>. {d} " for i, d in enumerate(new_docs) if any(word in d for word in key_nouns)])
                history_text = f"Old Question: {chat_history.get('question', '')}"
            else:
                question = content
                filtered_docs = 'None'
                history_text = f"Old Question: {chat_history.get('question', '')}\nOld Data: {chat_history.get('docs', '')}"
        else:
            if tok_query:
                question = ' '.join(tok_query)
                docs = ensemble_retriever.invoke(question)
                new_docs = list(set(doc.page_content.replace('\t', ' ') for doc in docs))
                if not new_docs:
                    raise NoDocumentsRetrievedError("No documents retrieved.")
                filtered_docs = " ".join([f"<Doc{i+1}>. {d} " for i, d in enumerate(new_docs) if any(word in d for word in key_nouns)])
                history_text = 'None'
            else:
                question = content
                filtered_docs = 'None'
                history_text = 'None'

        model = ChatOpenAI(
            streaming=True,
            verbose=True,
            callbacks=[callback],
            temperature=0.1,
            max_tokens=1500
        )

        year = datetime.now().year
        template = '''
        이 챗봇은 대구공업고등학교 100년사 책의 내용과 관련된 질문에 답변하는 안내원입니다. 답변은 한국어 높임말을 사용합니다.
        Read chat history to answer follow-up question.
        이름 옆의 괄호 내용를 이용해 동명이인을 구별하고, 질문에 연관된 동명이인은 모두 답변합니다.
        Answer the user's New Question using the following data. Individual docs may or may not be related to the question.
        Don't make up the answer.

        Year: {year}
        Chat history:
        {history_text}
        Data(fractions of book): {context}
        New Question: {question}
        New Answer:
        '''

        prompt = PromptTemplate(
            input_variables=[
                "year",
                "context",
                "question",
                "history_text"
            ],
            template=template
        )

        chain = prompt | model | StrOutputParser()
        
        task = asyncio.create_task(
            chain.ainvoke({"year": year, "context": filtered_docs, "question": content, "history_text": history_text})
        )

        async for token in callback.aiter():
            yield token

        response = await task
        chat_history['question'] = content
        chat_history['docs'] = filtered_docs
        global global_chat_history
        global_chat_history = chat_history
        
    except Exception as e:
        print(e)
        yield "죄송합니다. 지금은 답변해 드릴 수 없습니다."
    finally:
        callback.done.set()

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.post("/stream_chat/")
async def stream_chat(message: Message):
    generator = send_message(message.content, message.chat_history)
    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/get_chat_history/")
async def get_chat_history():
    return JSONResponse(content=global_chat_history)