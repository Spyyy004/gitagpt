import os
import time
import asyncio
import logging
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import MarianMTModel, MarianTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RAGApplication:
    def __init__(self, pdf_path: str = 'bgita.pdf', index_path: str = './vector_index'):
        logging.info("Initializing RAGApplication...")

        try:
            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings()
            logging.info("Loaded OpenAI embeddings.")

            # Load or create FAISS index
            self.vectorstore = self._load_or_create_index(pdf_path, index_path)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1, "search_type": "mmr"})
            logging.info("Vector store initialized successfully.")

            # Define prompt
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

            # Initialize ChatOpenAI model
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=512)
            logging.info("ChatOpenAI model initialized.")

            # Define RAG chain
            self.rag_chain = self.prompt | self.llm | StrOutputParser()

        except Exception as e:
            logging.error(f"Error initializing RAGApplication: {e}")
            raise

    def _load_or_create_index(self, pdf_path: str, index_path: str):
        try:
            if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
                logging.info("Loading existing FAISS index...")
                return FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            else:
                logging.info("FAISS index not found. Creating new index...")

                if not os.path.exists(index_path):
                    os.makedirs(index_path)

                # Load PDF
                logging.info(f"Loading PDF: {pdf_path}")
                loader = PyPDFLoader(pdf_path)
                docs_list = loader.load()

                if not docs_list:
                    logging.error("PDF loading failed! No documents found.")
                    raise ValueError("PDF is empty or could not be processed.")

                # Split text
                logging.info("Splitting text into chunks...")
                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=200, chunk_overlap=20
                )
                doc_splits = text_splitter.split_documents(docs_list)

                if not doc_splits:
                    logging.error("Text splitting failed! No chunks generated.")
                    raise ValueError("Text splitting failed.")

                # Create FAISS index
                logging.info(f"Creating FAISS index with {len(doc_splits)} document chunks...")
                vectorstore = FAISS.from_documents(doc_splits, self.embeddings)
                vectorstore.save_local(index_path)

                logging.info("FAISS index successfully created and saved.")
                return vectorstore

        except Exception as e:
            logging.error(f"Error in FAISS index creation: {e}")
            raise

    async def run(self, question: str) -> Dict:
        try:
            start_time = time.time()
            logging.info(f"Received question: {question}")

            logging.info("Fetching document embeddings from FAISS retriever...")
            
            # Debug FAISS retrieval
            try:
                documents = await self.retriever.ainvoke(question)
                logging.info(f"Retrieved {len(documents)} documents from FAISS.")
            except Exception as retrieval_error:
                logging.error(f"FAISS retrieval failed: {retrieval_error}")
                raise ValueError("FAISS retrieval failed. Index may be missing or corrupt.")

            retrieval_time = time.time() - start_time

            if not documents:
                logging.error("No documents retrieved! FAISS index might be empty or retrieval failed.")
                raise ValueError("Document retrieval failed. FAISS index may be missing or not properly loaded.")

            doc_texts = "\n".join([doc.page_content[:150] for doc in documents])

            logging.info("Calling LLM for response generation...")
            answer = await self.rag_chain.ainvoke({
                "question": question,
                "documents": doc_texts
            })

            llm_time = time.time() - start_time - retrieval_time
            logging.info(f"LLM response generated in {llm_time:.2f}s.")

            metadata = {
                "retrieved_documents": [doc.page_content for doc in documents],
                "num_documents": len(documents)
            }

            return {"answer": answer, "metadata": metadata}

        except ValueError as ve:
            logging.error(f"Retrieval Error: {ve}")
            raise HTTPException(status_code=500, detail=str(ve))

        except Exception as e:
            logging.error(f"Unexpected Error in run method: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

class Translator:
    def __init__(self):
        logging.info("Initializing Translator...")
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
        self.cache = {}

    async def translate_to_hindi(self, text: str) -> str:
        try:
            if text in self.cache:
                return self.cache[text]

            loop = asyncio.get_event_loop()
            translated_text = await loop.run_in_executor(None, self._translate_sync, text)
            self.cache[text] = translated_text
            logging.info("Translation completed.")
            return translated_text

        except Exception as e:
            logging.error(f"Translation error: {e}")
            raise

    def _translate_sync(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

# Initialize FastAPI
app = FastAPI()
rag = RAGApplication()
translator = Translator()

# Define API request and response models
class ChatRequest(BaseModel):
    message: str
    language: str = "English"

class ChatResponse(BaseModel):
    message: str
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        start_time = time.time()
        logging.info(f"API call received: {request.message} in {request.language}")

        if request.language.lower() == "hindi":
            rag_response = await rag.run(request.message)
            answer = await translator.translate_to_hindi(rag_response["answer"])
        else:
            rag_response = await rag.run(request.message)
            answer = rag_response["answer"]

        total_time = time.time() - start_time
        logging.info(f"Total response time: {total_time:.2f}s")

        return {"message": request.message, "answer": answer}

    except Exception as e:
        logging.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    logging.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
