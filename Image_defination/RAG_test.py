import json
import glob
import os
import re

def load_all_chunks(directory):
    all_chunks = []
    for filepath in glob.glob(os.path.join(directory, "chunks_*.json")):
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    return all_chunks

def retrieve_context_for_figure(chunks, figure_label):
    relevant_chunks = []
    
    # 1. Find where the figure appears & references
    # The figure caption usually contains "Figure X-X."
    caption_pattern = re.compile(rf"{figure_label}\.", re.IGNORECASE)
    ref_pattern = re.compile(rf"{figure_label}\b", re.IGNORECASE)
    
    # Group chunks by doc_id to allow neighboring chunks retrieval
    chunks_by_doc = {}
    for c in chunks:
        doc_id = c.get('doc_id')
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = {}
        chunks_by_doc[doc_id][c.get('chunk_index')] = c
        
    matched_indices = [] # list of (doc_id, chunk_index, type: 'caption' or 'reference')
    
    for c in chunks:
        text = c.get('text', '')
        if caption_pattern.search(text):
            matched_indices.append((c['doc_id'], c['chunk_index'], 'caption'))
        elif ref_pattern.search(text):
            matched_indices.append((c['doc_id'], c['chunk_index'], 'reference'))
            
    # Collect context
    collected_ids = set()
    collected_chunks = []
    
    def add_chunk(chunk):
        if chunk['chunk_id'] not in collected_ids:
            collected_ids.add(chunk['chunk_id'])
            collected_chunks.append(chunk)

    for doc_id, c_idx, m_type in matched_indices:
        doc_chunks = chunks_by_doc[doc_id]
        
        # Add the chunk itself
        if c_idx in doc_chunks:
            add_chunk(doc_chunks[c_idx])
            
        # If it's a caption, also get surrounding chunks
        if m_type == 'caption':
            if c_idx - 1 in doc_chunks:
                add_chunk(doc_chunks[c_idx - 1])
            if c_idx + 1 in doc_chunks:
                add_chunk(doc_chunks[c_idx + 1])
                
    # Rank & sort:
    # First: Surroundings of caption, then references
    # It's better to just sort by doc_id and chunk_index to preserve document order
    collected_chunks.sort(key=lambda x: (x['doc_id'], x['chunk_index']))
    
    return collected_chunks

if __name__ == "__main__":
    directory = "/home/saad/Desktop/Page_Index/Image_defination"
    chunks = load_all_chunks(directory)
    print(f"Loaded {len(chunks)} chunks.")
    
    target_figure = "Figure 5-8"
    ctx_chunks = retrieve_context_for_figure(chunks, target_figure)
    print(f"Retrieved {len(ctx_chunks)} chunks for {target_figure}.")
    
    for c in ctx_chunks:
        print("-----")
        print(f"Section: {c.get('section_title')}")
        print(c.get('text')[:200] + "...")
