import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from io import BytesIO

def extract_pages_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_data = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages_data.append({"page_content": text, "metadata": {"source": os.path.basename(pdf_path), "page": page_num + 1}})
    return pages_data

def chunk_pages(pages_data_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    all_chunks = []
    for page_data in pages_data_list:
        chunks_from_page = text_splitter.split_text(page_data["page_content"])
        
        for chunk_content in chunks_from_page:
            doc = Document(page_content=chunk_content, metadata=page_data["metadata"].copy())
            all_chunks.append(doc)
            
    return all_chunks

def get_pdf_page_image_bytes(pdf_path, page_number, highlight_texts=None):
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if not (0 <= page_number < len(doc)):
            if doc: doc.close()
            return None

        page = doc.load_page(page_number)
        
        if highlight_texts:
            for text_to_highlight in highlight_texts:
                try:
                    text_instances = page.search_for(text_to_highlight)
                    if text_instances:
                        for inst in text_instances:
                            highlight = page.add_highlight_annot(inst)
                except Exception as search_error:
                    print(f"Error while searching for text '{text_to_highlight}': {search_error}")

        pix = page.get_pixmap()
        img_bytes = BytesIO()
        pix.save(img_bytes, "png")
        img_bytes.seek(0)
        doc.close()
        return img_bytes.getvalue()
        
    except Exception as e:
        print(f"Error while retrieving page image ({pdf_path}, page {page_number}): {e}")
        if doc:
            doc.close()
        return None
    except Exception as e:
        print(f"Error while retrieving page image: {e}")
        if 'doc' in locals() and doc:
            doc.close()
        return None
