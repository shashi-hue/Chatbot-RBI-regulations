import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


os.environ["GOOGLE_API_KEY"] = "AIzaSyAOUkAsZoohJKMiwNXYLhmp0g1cqc2FxvQ"

pdf_path = "rbidocs.pdf"

print("Loading PDF...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("Building FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(splits, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)


template = """
You are a knowledgeable financial assistant specializing in RBI regulations and general finance.

You have access to both:
1. The RBI Master Direction document (context provided below)
2. Your general financial knowledge

When answering:
- Prefer facts from the document when available.
- If not in the document, use your general understanding to explain helpfully.
- Always mention if your explanation goes beyond the document.

Context:
{context}

Question: {question}

Answer clearly, in full sentences, and avoid repetition.
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    verbose=True,
)

def get_answer(query: str):
    """Helper function to answer a question using the RAG chain."""
    response = qa_chain.invoke({"query": query})
    return response["result"]

if __name__ == "__main__":
    print("\nRBI Chatbot. Type 'exit' to quit.\n")
    while True:
        query = input("Question: ")
        if query.lower() in ["exit", "quit"]:
            break
        print("\nAnswer:", get_answer(query), "\n")

