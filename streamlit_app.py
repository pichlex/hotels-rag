import streamlit as st
import os
import lancedb
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
import uuid
from dotenv import load_dotenv

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set OPENAI_API_KEY in .env file")

def initialize_components():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    db = lancedb.connect("lancedb/hotels_rag_db")
    
    vectorstore = LanceDB(
        connection=db,
        embedding=embeddings,
        table_name="embeddings"
    )
    
    return vectorstore.as_retriever(), ChatOpenAI(model="gpt-4.1")

def create_rag_chain(retriever, llm):
    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Ты - профессиональный специалист по бронированию отелей в туристическом агентстве. "
         "Будь дружелюбен, внимателен к деталям и всегда стремись превзойти ожидания клиента. "
         "Используй всю доступную историю чата, чтобы делать персонализированные и точные предложения. "
         "Твоя задача - подобрать для клиента лучшие варианты отелей, учитывая его пожелания и предыдущие вопросы. "
         "Если потребуется дополнительная информация, вежливо уточни детали. "
         "Помни: качественный сервис - залог твоей годовой премии!"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("system", "Начни поиск подходящих отелей, учитывая пожелания клиента и историю диалога.")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Используй следующий контекст для ответа на вопрос клиента о бронировании отелей:\n"
         "{context}\n"
         "---\n"
         "История чата:\n"
         "{chat_history}\n"
         "Дай чёткий, полезный и лаконичный ответ, ссылаясь на релевантную информацию из контекста. "
         "Если информации недостаточно, вежливо уточни детали у клиента."),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, history_aware_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


#--------Streamlit UI------------

st.title("🏨 Ассистент")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

retriever, llm = initialize_components()
rag_chain = create_rag_chain(retriever, llm)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Спросите что-нибудь..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = conversational_rag_chain.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        answer = response["answer"]
        
        st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
