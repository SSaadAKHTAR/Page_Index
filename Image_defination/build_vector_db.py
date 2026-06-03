import json
import re
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def parse_markdown_to_chunks(md_filepath, chunk_size=800, overlap=100):
    with open(md_filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split roughly by paragraphs or headers
    raw_chunks = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for block in raw_chunks:
        block = block.strip()
        if not block:
            continue
            
        if len(current_chunk) + len(block) < chunk_size:
            current_chunk += "\n\n" + block if current_chunk else block
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Simple overlap handling (rough approximation)
            words = current_chunk.split()
            overlap_words = words[-overlap:] if len(words) > overlap else []
            current_chunk = " ".join(overlap_words) + "\n\n" + block

    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def build_vector_db():
    md_filepath = "/home/saad/Desktop/Page_Index/Image_defination/output_hierarchical.md"
    db_out_dir = "/home/saad/Desktop/Page_Index/Image_defination"
    
    print(f"Reading {md_filepath}...")
    if not os.path.exists(md_filepath):
         print(f"Error: {md_filepath} does not exist.")
         return
         
    chunks = parse_markdown_to_chunks(md_filepath, chunk_size=1000, overlap=50)
    print(f"Generated {len(chunks)} chunks.")
    
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    print("Adding vectors to FAISS index...")
    index.add(embeddings)
    
    # Save index
    faiss_index_path = os.path.join(db_out_dir, "faiss_index.bin")
    faiss.write_index(index, faiss_index_path)
    
    # Save metadata
    metadata_path = os.path.join(db_out_dir, "faiss_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump([{"chunk_index": i, "text": c} for i, c in enumerate(chunks)], f, indent=4)
        
    print(f"Saved FAISS index to {faiss_index_path}")
    print(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    build_vector_db()
