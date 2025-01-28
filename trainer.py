import asyncio
import os
import time
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from transformers import MarianMTModel, MarianTokenizer

# Define custom embedding class
class CustomEmbedding(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=32, show_progress_bar=False).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

# Define the RAG application class
class RAGApplication:
    def __init__(self, pdf_path: str = 'bgita.pdf', index_path: str = './vector_index'):
        self.embeddings = CustomEmbedding()

        # Load or create the FAISS index during initialization
        self.vectorstore = self._load_or_create_index(pdf_path, index_path)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 1, "search_type": "mmr"}  # Optimize retrieval
        )

        # Define the prompt template
        self.prompt = PromptTemplate(
            template="""BHAKTI CONTEXT GUIDELINES:
            - You are Lord Krishna yourself interpreting the Bhagavad Gita.
            - Provide profound, compassionate insights rooted in the Gita's teachings.
            - Keep your answer concise (maximum 4-5 sentences).
            - Reference the verse(s) from which your answer is derived.

            CONTEXT: {documents}
            QUESTION: {question}

            RESPONSE REQUIREMENTS:
            - Deeply philosophical
            - Practically applicable
            - Concise yet profound

            ANSWER:""",
            input_variables=["question", "documents"]
        )

        # Initialize the LLM
        self.llm = ChatOllama(
            model="llama3.2:3b-instruct-fp16",  # Use a smaller model
            temperature=0.3,
            num_ctx=512
        )

        # Define the RAG chain
        self.rag_chain = self.prompt | self.llm | StrOutputParser()

    def _load_or_create_index(self, pdf_path: str, index_path: str):
        if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
            return FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            if not os.path.exists(index_path):
                os.makedirs(index_path)

            # Load the PDF and split the text
            loader = PyPDFLoader(pdf_path)
            docs_list = loader.load()

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=200,
                chunk_overlap=20
            )
            doc_splits = text_splitter.split_documents(docs_list)

            # Create the FAISS index
            vectorstore = FAISS.from_documents(doc_splits, self.embeddings)
            vectorstore.save_local(index_path)
            return vectorstore

    async def run(self, question: str) -> Dict:
        start_time = time.time()

        # Use async retrieval
        documents = await self.retriever.ainvoke(question)
        retrieval_time = time.time() - start_time
        print(f"Retrieval time: {retrieval_time:.2f}s")

        doc_texts = "\n".join([doc.page_content[:150] for doc in documents])

        # Use async chain invocation
        answer = await self.rag_chain.ainvoke({
            "question": question,
            "documents": doc_texts
        })
        llm_time = time.time() - start_time - retrieval_time
        print(f"LLM inference time: {llm_time:.2f}s")

        metadata = {
            "retrieved_documents": [doc.page_content for doc in documents],
            "num_documents": len(documents)
        }

        return {"answer": answer, "metadata": metadata}

# Initialize translation utilities
class Translator:
    def __init__(self):
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
        self.cache = {}  # Cache for translations

    async def translate_to_hindi(self, text: str) -> str:
        if text in self.cache:
            return self.cache[text]

        loop = asyncio.get_event_loop()
        # Run the synchronous translation in a separate thread
        translated_text = await loop.run_in_executor(None, self._translate_sync, text)
        self.cache[text] = translated_text
        return translated_text

    def _translate_sync(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

# Initialize the FastAPI app
app = FastAPI()

# Initialize the RAG system and translator
rag = RAGApplication()
translator = Translator()

# Define the input schema for the API
class ChatRequest(BaseModel):
    message: str
    language: str = "English"  # Default to English

# Define the output schema for the API
class ChatResponse(BaseModel):
    message: str
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        start_time = time.time()

        if request.language.lower() == "hindi":
            # For Hindi, use the async pipeline
            rag_response = await rag.run(request.message)
            answer = await translator.translate_to_hindi(rag_response["answer"])
        else:
            # For English, use the async pipeline
            rag_response = await rag.run(request.message)
            answer = rag_response["answer"]

        total_time = time.time() - start_time
        print(f"Total response time: {total_time:.2f}s")

        # Construct the response
        response = {
            "message": request.message,
            "answer": answer
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)