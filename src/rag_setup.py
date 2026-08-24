import os

import chromadb

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

import pypdf



def extract_text_from_pdf(pdf_path):

    print(f"Extracting text from {pdf_path}...")

    text = ""

    try:

        with open(pdf_path, 'rb') as f:

            reader = pypdf.PdfReader(f)

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    text += page_text + "\n"

    except Exception as e:

        print(f"Error reading {pdf_path}: {e}")

    return text



def setup_rag():

    print("Setting up RAG Vector Database...")

    db_dir = 'chroma_db'

    if not os.path.exists(db_dir):

        os.makedirs(db_dir)



    client = chromadb.PersistentClient(path=db_dir)

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


    try:

        client.delete_collection(name="pump_manuals")

    except ValueError:

        pass

        

    collection = client.create_collection(name="pump_manuals")



    manuals_dir = 'manuals'

    

    md_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)



    docs = []

    metadatas = []

    ids = []

    doc_id_counter = 0



    if not os.path.exists(manuals_dir):

        print(f"Error: {manuals_dir} not found.")

        return



    for filename in os.listdir(manuals_dir):

        filepath = os.path.join(manuals_dir, filename)

        

        if filename.endswith(".md"):

            print(f"Processing markdown: {filename}")

            with open(filepath, 'r', encoding='utf-8') as f:

                content = f.read()

            chunks = md_splitter.split_text(content)

            

        elif filename.endswith(".pdf"):

            content = extract_text_from_pdf(filepath)

            chunks = text_splitter.split_text(content)

            

        else:

            continue

            

        for i, chunk in enumerate(chunks):

            docs.append(chunk)

            metadatas.append({"source": filename, "chunk_id": i})

            ids.append(f"doc_{doc_id_counter}")

            doc_id_counter += 1



    if len(docs) == 0:

        print("No documents to embed.")

        return



    print(f"Embedding {len(docs)} chunks from {len(os.listdir(manuals_dir))} files...")

    batch_size = 1000

    for i in range(0, len(docs), batch_size):

        batch_docs = docs[i:i+batch_size]

        batch_meta = metadatas[i:i+batch_size]

        batch_ids = ids[i:i+batch_size]

        batch_emb = embedding_model.encode(batch_docs).tolist()

        

        collection.add(

            documents=batch_docs,

            embeddings=batch_emb,

            metadatas=batch_meta,

            ids=batch_ids

        )

        print(f"Inserted batch {i//batch_size + 1}")

        

    print("RAG Vector Database setup complete!")



if __name__ == "__main__":

    setup_rag()

