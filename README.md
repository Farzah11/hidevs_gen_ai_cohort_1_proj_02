# hidevs_gen_ai_cohort_1_proj_02
**Question Answering System with RAG -**

!pip install langchain-community
!pip install langchain pymupdf
!pip install langchain-pinecone
!pip install tiktoken
!pip install openpyxl
!pip install python-docx

import os
import pandas as pd
from langchain.document_loaders import PyMuPDFLoader # This should now work
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import pinecone as lang_pinecone
from langchain_pinecone import PineconeVectorStore as lang_pinecone
# library to keep sensitive info in a separate .env file
from dotenv import load_dotenv
import os
import tiktoken
goq_api_key= insert your key here
PINECONE_API_KEY=insert your key here
PINECONE_INDEX_NAME="rag-model"
# to access the info in .env file
pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('PINECONE_INDEX_NAME')

**#MASTER FUNCTION THAT WILL AUTOMATICALLY SWITCH TO THE LOADER ASKED BY USER**
# loading and splitting
import os

def load_document(file_path):
    """
    Master function that selects the appropriate document loader based on file type.
    :param file_path: Path to the document file
    :return: Loaded document content and split into chunks
    """
    # Get the file extension from the provided file_path
    file_extension = os.path.splitext("/content/Book1.xlsx")[1].lower()

    print(f"Loading document: {/content/Book1.xlsx}")

    if file_extension == ".pdf":
        print("PDF file detected. Using PDF loader.")
        return load_pdf("/content/National AI Policy Consultation Draft V1.pdf")
    elif file_extension == ".csv":
        print("CSV file detected. Using CSV loader.")
        return load_csv("/content/csv_fileformat.csv")
    elif file_extension == ".docx":
        print("DOCX file detected. Using DOCX loader.")
        return load_docx("/content/Asha_journey.docx")
    elif file_extension in [".xlsx", ".xls"]:
        print("Excel file detected. Using Excel loader.")
        return load_excel("/content/Book1.xlsx")
    else:
        raise ValueError("Unsupported file type. Please provide a PDF, CSV, DOCX, or Excel file.")
# Subfunction for loading and splitting PDFs
def load_pdf(file_path):
    reader = PyMuPDFLoader("/content/National AI Policy Consultation Draft V1.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Split the text using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    return chunks

# Subfunction for loading and splitting CSV files
def load_csv(file_path):
    df = pd.read_csv("/content/csv_fileformat.csv")

    # Split the DataFrame into chunks (e.g., by rows)
    row_chunks = [df.iloc[i:i + 50] for i in range(0, len(df), 50)]

    return row_chunks

# Subfunction for loading and splitting DOCX files
def load_docx(file_path):
    doc = Document("/content/Asha_journey.docx")
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    # Split the text using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    return chunks

# Subfunction for loading and splitting Excel files
def load_excel(file_path):
    workbook = load_workbook("/content/Book1.xlsx")
    sheet = workbook.active

    # Convert the sheet into a DataFrame
    data = sheet.values
    df = pd.DataFrame(data)

    # Split the DataFrame into chunks (similar to CSV)
    row_chunks = [df.iloc[i:i + 50] for i in range(0, len(df), 50)]

    return row_chunks
# EMBEDDINGS
    
    
    def embed_and_store(chunks):
    """
    Function to embed document chunks using GrokEmbeddings and store them in Pinecone.
    :param chunks: List of document chunks to be embedded
    """
    embeddings = [grok_embeddings.embed(chunk) for chunk in chunks]

    # Upsert the embeddings into Pinecone
    ids = [str(i) for i in range(len(embeddings))]  # Generate unique IDs for each chunk
    vectorstore.upsert(vectors=zip(ids, embeddings))

    print(f"{len(embeddings)} document chunks embedded and stored in Pinecone.")
# VECTOR_SRORE
    

    from typing import List, Any

def multi_query_retrieval(vector_store, query: str, k: int = 5) -> List[Any]:
    """
    Retrieve relevant chunks based on multiple variations of the user query.
    :param vector_store: The vector store instance
    :param query: The user query
    :param k: Number of top results to retrieve
    :return: Relevant document chunks
    """
    query_variations = generate_query_variations(query)
    all_results = []
    for var in query_variations:
        results = vector_store.similarity_search(var, k=k)
        all_results.extend(results)
    # Remove duplicates and sort by relevance
    return sorted(list(set(all_results)), key=lambda x: x.metadata['score'], reverse=True)[:k]
# LARGE LANGUAGE MODEL(LLM)
    
    !pip install groq
    import groq  # Import the Groq library (ensure it's installed)

def setup_llm(api_key: str):
    """
    Set up the Groq language model with API key.
    :param api_key: API key for authenticating with Groq
    :return: A Groq pipeline for text generation
    """
    # Initialize Groq client with the API key
    groq_client = groq.Client(api_key= insert your key here)  # Replace with actual client initialization

    # Load the Groq model (adjust based on the actual Groq library/API)
    model_name = "proj-name"  # Replace with the actual model name or path
    model = groq_client.load_model(proj)  # Example; replace with actual loading mechanism

    # Create a pipeline for text generation (adjust based on the Groq API)
    pipe = groq.Pipeline(
        model=model,
        max_length=2048,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    return pipe

  # Q/A chain(prompt_template)
    def rag_pipeline(query: str, vector_store, llm):
    """
    Execute the RAG pipeline for a given query.
    :param query: The user query
    :param vector_store: The vector store instance
    :param llm: The language model instance
    :return: The result of the query
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:"""
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa({"query": query})


    if __name__ == "__main__":
    file_path = "path/to/your/document.pdf"  # Replace with your document path
    query = "What is the main topic of the document?"  # Replace with your query
    pinecone_api_key = "your-pinecone-api-key"  # Replace with your Pinecone API key
    pinecone_env = "your-pinecone-environment"  # Replace with your Pinecone environment
    pinecone_index_name = "your-pinecone-index-name"  # Replace with your Pinecone index name

    main(file_path, query, pinecone_api_key, index_name)


