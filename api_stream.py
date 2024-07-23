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
        if len(tok_query) == 0:
            tok_query = chat_history[-1]['tok_question']
            question = content + ' ' + ' '.join(tok_query)
        else:
            question = ' '.join(tok_query)
        #print(question)

        docs = ensemble_retriever.invoke(question)
        new_docs = list(set(doc.page_content.replace('\t', ' ') for doc in docs))
        filtered_docs = [f"<Doc{i+1}>. {d}" for i, d in enumerate(new_docs) if any(word in d for word in tok_query)]
        #print(filtered_docs)
        
        if not new_docs:
          raise NoDocumentsRetrievedError("No documents retrieved.")
        
        model = ChatOpenAI(
            streaming=True,
            verbose=True,
            callbacks=[callback],
            temperature=0
        )

        year = datetime.now().year
        history = "\n".join(f"Q: {item['question']}\nA(참고): {item['answer']}" for item in chat_history[-2:])
        #print(history)
        template = '''
        너는 대구공업고등학교 80년사 책의 내용과 관련된 질문에 답변하는 챗봇이야. 답변은 한국어 높임말로 써. 일관되고 친절한 말투를 써.
        데이터에는 질문과 관련없는 내용도 포함되어 있기 때문에 질문에 대한 확실한 답만을 찾아 답변해줘. 특정 사람과 관련된 질문이면 성과 이름이 "완벽히" 일치하는 자료를 찾아. 같은 이름을 가진 사람이 있어도 동명이인일 수 있기 떄문에 디테일을 보고 구별해서 알려줘(회수, 기수, 분야, 직업). 올해는 {year}년이야.
        DON'T MAKE UP THE ANSWER. Don't repeat the answer from chat history.
        Data(책 내용의 일부, don't connect docs with different numbers.): {context}
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
            chain.ainvoke({"year": year, "context": filtered_docs, "question": content, "history": history})
        )

        async for token in callback.aiter():
            yield token

        response = await task
        chat_history.append({"question": content, "tok_question": tok_query, "answer": response})
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