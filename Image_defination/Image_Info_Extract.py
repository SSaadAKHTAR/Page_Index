from mistralai.client import Mistral
import base64
import os
import glob
import re
import json
import faiss
from sentence_transformers import SentenceTransformer

MODEL = "pixtral-12b-2409"
API_KEY = "85rLAhDO5zwXHBFSkOWAa03OQcos5lPs"

def load_vector_db():
    db_out_dir = "/home/saad/Desktop/Page_Index/Image_defination"
    faiss_index_path = os.path.join(db_out_dir, "faiss_index.bin")
    metadata_path = os.path.join(db_out_dir, "faiss_metadata.json")
    
    if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
        print("Vector database files not found. Skipping RAG.")
        return None, None
        
    print("Loading FAISS index...")
    index = faiss.read_index(faiss_index_path)
    
    print("Loading metadata...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    return index, metadata

def retrieve_rag_context(query, embedding_model, index, metadata, top_k=5):
    if index is None or metadata is None:
        return ""
    
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, top_k)
    
    context_chunks = []
    for i in indices[0]:
        if i >= 0 and i < len(metadata):
            context_chunks.append(metadata[i]["text"])
            
    return "\n\n---\n\n".join(context_chunks)

def extract_image_definition(client, image_path, image_filename, rag_context):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    print(f"Extracting definition using API for: {image_filename}")
    
    prompt_text = (
        f"The image title is {image_filename}. Create a robust, specification-aware definition for this figure.\n\n"
        f"Use the following textual context retrieved from the specification to inform and enrich your definition, "
        f"and use the specification terminology accurately:\n\n"
        f"--- RAG CONTEXT ---\n{rag_context}\n--- END RAG CONTEXT ---\n\n"
        f"Carefully examine the image to validate the text. Explain visible components, signals, interfaces, data flows, "
        f"states, and connections. Do not hallucinate components that aren't present in either the context or the image."
    )
    
    response = client.chat.complete(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_b64}",
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

def insert_definition_into_md(md_path, figure_label, definition):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    inserted = False
    for i, line in enumerate(lines):
        if line.startswith(f"{figure_label}.") or line.startswith(f"{figure_label} :"):
            # Insert the definition right after this line
            lines.insert(i + 1, f"\n> **Generated Definition for {figure_label}**: {definition.strip()}\n\n")
            print(f"Inserted definition for '{figure_label}' at line {i + 1}.")
            inserted = True
            break
            
    if not inserted:
        print(f"Warning: Figure label '{figure_label}' not found at the start of any line in markdown.")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def main():
    client = Mistral(api_key=API_KEY)
    
    # RAG Setup
    print("Loading sentence transformer model for RAG...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index, metadata = load_vector_db()
    
    image_dir = "/home/saad/Desktop/Page_Index/Image_defination/Images"
    # Keeping output markdown since it's hardcoded as UCIE_1.1.md originally
    # wait, they requested output_hierarchical.md as the doc but they didn't specify where to write definitions. Let's ask or write to output_hierarchical.md.
    # We will write back to output_hierarchical.md instead of UCIE_1.1.md. 
    md_path = "/home/saad/Desktop/Page_Index/Image_defination/output_hierarchical.md"
    
    png_files = glob.glob(os.path.join(image_dir, "*.png"))
    png_files.sort()
    
    count = 0
    for img_path in png_files:
        if count >= 10:
            print("Reached 10 images limit. Stopping.")
            break
            
        filename = os.path.basename(img_path)
        match = re.match(r"(Figure \d+-\d+)\.?", filename)
        if match:
            figure_label = match.group(1)
            
            # Fetch RAG Context
            context_query = figure_label
            rag_context = retrieve_rag_context(context_query, embedding_model, faiss_index, metadata)
            
            # Extract Image Definition
            definition = extract_image_definition(client, img_path, filename, rag_context)
            
            # Insert Definition
            insert_definition_into_md(md_path, figure_label, definition)
            count += 1
        else:
            print(f"Skipping {filename}: Could not extract a matching 'Figure X-X.' label.")

if __name__ == "__main__":
    main()