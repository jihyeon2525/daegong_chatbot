from unstruct_step1 import Vec, documents
import asyncio
from typing import AsyncIterable
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote.retrievers import KiwiBM25Retriever
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
from kiwipiepy import Kiwi
from datetime import datetime
import os

MAX_CHAT_HISTORY = 3

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
def noun_extractor(text):
    results = []
    result = kiwi.analyze(text)
    for token, pos, _, _ in result[0][0]:
        if len(token) != 1 and pos=='NNG' or pos=='NNP' or pos=='NR' or pos=="XR" or pos=='SL' or pos=='SH' or pos=='SN':
            results.append(token)
    return results

kbm25_retriever = KiwiBM25Retriever.from_documents(documents, k=4)
embed_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
ensemble_retriever = EnsembleRetriever(retrievers=[kbm25_retriever, embed_retriever], weights=[0.6, 0.4])

class Message(BaseModel):
    content: str

class NoDocumentsRetrievedError(Exception):
    pass

chat_history = []
async def send_message(content: str) -> AsyncIterable[str]:
    try:
        callback = AsyncIteratorCallbackHandler()
        
        tok_query = noun_extractor(content)
        if chat_history:
            if tok_query:
                question = ' '.join(tok_query)
                docs = ensemble_retriever.invoke(question)
                new_docs = list(set(doc.page_content.replace('\t', ' ') for doc in docs))
                if not new_docs:
                    raise NoDocumentsRetrievedError("No documents retrieved.")
                filtered_docs = [f"<Doc{i+1}>. {d}" for i, d in enumerate(new_docs) if any(word in d for word in tok_query)]
                history = "\n".join(f"Old Question: {item['question']}\nOld Answer: {item['answer']}" for item in chat_history[-1:])
                print("history 있고 키워드 있음")
            else:
                question = content
                filtered_docs = 'None'
                history = "\n".join(f"Old Question: {item['question']}\nOld Retrieved Data: {item['docs']}" for item in chat_history[-1:])
                print("history 있고 키워드 없음")
        else:
            if tok_query:
                question = ' '.join(tok_query)
                docs = ensemble_retriever.invoke(question)
                new_docs = list(set(doc.page_content.replace('\t', ' ') for doc in docs))
                if not new_docs:
                    raise NoDocumentsRetrievedError("No documents retrieved.")
                filtered_docs = [f"<Doc{i+1}>. {d}" for i, d in enumerate(new_docs) if any(word in d for word in tok_query)]
                history = 'None'
                print("history 없고 키워드 있음")
            else:
                question = content
                filtered_docs = 'None'
                history = 'None'
                print("history 없고 키워드 없음")
        #print(tok_query)
        #print(question)
        #print(filtered_docs)
        #print(history)
        
        model = ChatOpenAI(
            streaming=True,
            verbose=True,
            callbacks=[callback],
            temperature=0.1
        )

        year = datetime.now().year
        template = '''
        이 챗봇은 대구공업고등학교 100년사 책의 내용과 관련된 질문에 답변하는 안내원입니다. 답변은 한국어 "높임말"로 합니다.
        Read Retrieved Data and Chat history before answering question. DON'T MAKE UP THE ANSWER.
        Never repeat the Old Answer from chat history. 
        특정 사람과 관련된 질문이면 이름, 회수, 기수, 분야(전기, 기계, 방직, 화학, 화공, 건축, 자동차, 토목, 섬유), 직업 등을 보고 엄격하게 구분합니다. (예를 들어, 이진호(섬유)와 이진호(방직)은 다른 사람입니다.) 올해는 {year}년입니다.
        
        Retrieved Data(fractions of the book): {context}
        New Question: {question}
        Chat history:
        {history}
        New Answer:
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
            chain.ainvoke({"year": year, "context": filtered_docs, "question": content, "history": history})
        )

        async for token in callback.aiter():
            yield token

        response = await task
        chat_history.append({"question": content, "tok_question": tok_query, "docs": filtered_docs, "answer": response})
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