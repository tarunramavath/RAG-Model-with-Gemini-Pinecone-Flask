from flask import Flask,request,jsonify,render_template,session
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold,HarmCategory
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

app=Flask(__name__)

app.secret_key=os.urandom(24)

def initialize_qa_system():
    loader=TextLoader('state_union.txt',encoding="utf-8")
    document=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts=text_splitter.split_documents(document)

    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMENI_API_KEY"),
        task_type="retrieval_query" 
    )

    os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")

    vectordb=PineconeVectorStore.from_documents(texts,embeddings,index_name=os.getenv("PINECONE_INDEX_NAME"))

    Prompt_template="""
    ## Safety and Respect Come First!

    You are programmed to be a helpful and harmless AI. You will not answer requests that promote:

    * Harassment or Bullying: Targeting individuals or groups with hateful or hurtful language.
    * Hate Speech:  Content that attacks or demeans others based on race, ethnicity, religion, gender, sexual orientation, disability, or other protected characteristics.
    * Violence or Harm:  Promoting or glorifying violence, illegal activities, or dangerous behavior.
    * Misinformation and Falsehoods:  Spreading demonstrably false or misleading information.

    How to Use You:

    1. Provide Context: Give me background information on a topic.
    2. Ask Your Question: Clearly state your question related to the provided context.

    ##  Answering User Question:
    Context: \n {context}
    Question: \n {question}
    Answer:
    """

    prompt=PromptTemplate(template=Prompt_template,input_variables=["context","questions"])

    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    chat_model=ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMENI_API_KEY"),
        temperature=0.7,
        safety_settings=safety_settings
    )

    retriever_from_llm=MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k":5}),
        llm=chat_model
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever_from_llm,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

qa_chain=initialize_qa_system()

def get_conversation_history():
    if "conversation" not in session:
        session['conversion']=[]
    return session['conversion']

def update_conversion_history(user_question,bot_respponce):
    conversation=get_conversation_history()
    conversation.append({"user":user_question,"bot":bot_respponce})
    session['conversion']=conversation

@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        user_question = request.form["question"]
        context = "document context here"  
        response = qa_chain.invoke({"query": user_question, "context": context})
        bot_response = response['result']  
        source_documents = response.get('source_documents', [])
        source_texts = [doc.page_content for doc in source_documents]
        return render_template("index.html", user_question=user_question, bot_response=bot_response, source_documents=source_texts)

    return render_template("index.html", user_question=None, bot_response=None)

if __name__ == "__main__":
    app.run(debug=True)
