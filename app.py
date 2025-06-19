import streamlit as st
import os
from pdf_handler import extract_pages_from_pdf, chunk_pages, get_pdf_page_image_bytes
import fitz
from embedder import embed_and_store, load_vectorstore
from chatbot import get_qa_chain, generate_suggested_questions, summarize_documents, extract_keywords_from_documents, generate_concept_map_data, extract_timeline_from_documents
import json
import pandas as pd 
from collections import defaultdict 

st.set_page_config(page_title="PDF Chatbot", layout="wide")


if 'pdf_previews' not in st.session_state:
    st.session_state.pdf_previews = {}
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = ""
if 'refined_answer' not in st.session_state:
    st.session_state.refined_answer = ""
if 'source_documents' not in st.session_state:
    st.session_state.source_documents = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []
if 'current_question_input' not in st.session_state:
    st.session_state.current_question_input = ""
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = ""
if 'extracted_keywords' not in st.session_state:
    st.session_state.extracted_keywords = []
if 'concept_map_data' not in st.session_state:
    st.session_state.concept_map_data = ""
if 'timeline_data' not in st.session_state:
    st.session_state.timeline_data = ""
if 'page_chunk_counts' not in st.session_state: 
    st.session_state.page_chunk_counts = {}


st.title("📄 PDF-Supported Role-Based Chatbot")

st.info(
    "🔒 **Data Privacy and Security:** This application runs entirely on your local machine. "
    "The PDFs you upload and the questions you ask are never sent to any external server. "
    "All processes — text extraction, embedding, and answer generation — are performed locally on your computer using Ollama and your gemma:2b model."
)
st.markdown("---")


roles_data = json.load(open("roles.json", "r", encoding="utf-8"))
available_roles = roles_data
selected_role_from_list = st.selectbox("🧑 Choose from Predefined Roles", [""] + available_roles, index=0, help="Select a role or enter your own below.")

custom_role_input = st.text_area("📝 Or Write Your Own Role (optional)", placeholder="Example: A teacher who analyzes education-related issues in documents.")


final_selected_role = custom_role_input.strip() if custom_role_input.strip() else selected_role_from_list

if not final_selected_role:
    st.warning("⚠️ Please select a role or write your own.")
    st.stop()

st.info(f"🤖 Select Role: {final_selected_role}")


available_languages = {"Hindi": "Hi", "English": "en"}
selected_language_label = st.selectbox("🌐 Select response language", list(available_languages.keys()))
selected_language_code = available_languages[selected_language_label]


uploaded_files = st.file_uploader("📤 Upload PDF (You can select multiple files)", type=["pdf"], accept_multiple_files=True)
processed_pdf_paths = []
all_chunks_for_session = [] 

if uploaded_files:

    st.session_state.pdf_previews = {}
    st.session_state.suggested_questions = []
    st.session_state.document_summary = ""
    st.session_state.extracted_keywords = []
    st.session_state.concept_map_data = ""
    st.session_state.timeline_data = ""
    st.session_state.page_chunk_counts = {}
    st.session_state.last_answer = ""
    st.session_state.refined_answer = ""
    st.session_state.source_documents = []

    current_file_chunks = []
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        processed_pdf_paths.append(file_path)

        try:
            doc = fitz.open(file_path)
            st.session_state.pdf_previews[uploaded_file.name] = {
                "total_pages": doc.page_count,
                "current_page_display": 1,
                "path": file_path
            }
            doc.close()
        except Exception as e:
            st.error(f"{uploaded_file.name} Page count could not be retrieved: {e}")
            continue

        pages_data = extract_pages_from_pdf(file_path)
        if pages_data:
            chunks_from_file = chunk_pages(pages_data)
            current_file_chunks.extend(chunks_from_file)
            st.write(f"📄 {uploaded_file.name} ({st.session_state.pdf_previews.get(uploaded_file.name, {}).get('total_pages', 'N/A')} pages)  processed ({len(chunks_from_file)} chunk).")
        else:
            st.write(f"⚠️ {uploaded_file.name} Text could not be extracted from the file or the file is empty.")

    all_chunks_for_session = current_file_chunks

    if all_chunks_for_session:

        page_counts = defaultdict(lambda: defaultdict(int))
        for chunk in all_chunks_for_session:
            source = chunk.metadata.get("source", "Unknown Source")
            page = chunk.metadata.get("page", 0)
            if page > 0: 
                page_counts[source][page] += 1
        st.session_state.page_chunk_counts = {k: dict(v) for k, v in page_counts.items()} 


        if not os.path.exists("vectordb"):
            os.makedirs("vectordb")
        embed_and_store(all_chunks_for_session)
        st.success("✅ All PDFs have been processed and the database has been created/updated!")

        with st.spinner("🤔 Sample questions and keywords are being prepared...."):
            st.session_state.suggested_questions = generate_suggested_questions(all_chunks_for_session, final_selected_role, selected_language_code, num_questions=3)
            st.session_state.extracted_keywords = extract_keywords_from_documents(all_chunks_for_session, final_selected_role, selected_language_code, num_keywords=10)
    else:
        st.warning("⚠️ Text could not be extracted from the uploaded PDFs or the PDFs are empty.")
        st.session_state.suggested_questions = []
        st.session_state.extracted_keywords = []
        st.session_state.concept_map_data = ""
        st.session_state.timeline_data = ""
        st.session_state.page_chunk_counts = {}



if st.session_state.pdf_previews:
    st.markdown("---")
    st.subheader("📂 Uploaded PDFs and Preview.")
    for pdf_name, preview_data in st.session_state.pdf_previews.items():
        with st.expander(f"{pdf_name} ({preview_data['total_pages']} pages)"):
            if preview_data['total_pages'] > 0:
                page_to_show_user = st.number_input(
                    f"pages numbers (1-{preview_data['total_pages']})",
                    min_value=1,
                    max_value=preview_data['total_pages'],
                    value=preview_data['current_page_display'],
                    key=f"preview_page_num_{pdf_name}"
                )
                st.session_state.pdf_previews[pdf_name]['current_page_display'] = page_to_show_user
                page_num_fitz = page_to_show_user - 1

                texts_to_highlight_on_page = []
                if st.session_state.source_documents:
                    for src_doc in st.session_state.source_documents:
                        if src_doc.metadata.get('source') == pdf_name and \
                           src_doc.metadata.get('page') == page_to_show_user:
                            texts_to_highlight_on_page.append(src_doc.page_content)

                img_bytes = get_pdf_page_image_bytes(preview_data['path'], page_num_fitz, texts_to_highlight_on_page)
                if img_bytes:
                    st.image(img_bytes, caption=f"{pdf_name} - page {page_to_show_user}{' (highlighted)' if texts_to_highlight_on_page else ''}", use_column_width=True)
                else:
                    st.warning(f"{pdf_name} - page {page_to_show_user} Preview could not be generated.")
            else:
                st.info(f"{pdf_name} The content is empty or unreadable.")
if st.session_state.page_chunk_counts:
    st.markdown("---")
    st.subheader("📊 Information Density (Chunks per Page)")
    for pdf_name, counts in st.session_state.page_chunk_counts.items():
        with st.expander(f"{pdf_name} - Density Chart"):
            if counts:
                sorted_counts = dict(sorted(counts.items()))
                st.bar_chart(sorted_counts)
            else:
                st.info(f"Density data could not be calculated for {pdf_name}.")

if uploaded_files and all_chunks_for_session:
    if st.session_state.extracted_keywords:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔑 Keywords")
        for kw in st.session_state.extracted_keywords:
            st.sidebar.caption(kw)

    if st.session_state.suggested_questions:
        st.markdown("---")
        st.subheader("💡 Example Questions")
        num_suggestion_cols = min(len(st.session_state.suggested_questions), 3)
        if num_suggestion_cols > 0:
            cols = st.columns(num_suggestion_cols)
            for i, sq in enumerate(st.session_state.suggested_questions):
                with cols[i % num_suggestion_cols]:
                    if st.button(sq, key=f"suggested_q_{i}", use_container_width=True):
                        st.session_state.current_question_input = sq
                        st.experimental_rerun()

    question = st.text_input("❓ Ask a Question", value=st.session_state.current_question_input, key="main_question_input_field")
    if st.session_state.main_question_input_field != st.session_state.current_question_input:
        st.session_state.current_question_input = st.session_state.main_question_input_field
        st.experimental_rerun()

    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("💬 Answer", use_container_width=True) and st.session_state.current_question_input:
            st.session_state.last_question = st.session_state.current_question_input
            st.session_state.refined_answer = ""
            st.session_state.document_summary = ""
            st.session_state.concept_map_data = ""
            st.session_state.timeline_data = ""
            vectorstore = load_vectorstore()
            if vectorstore:
                with st.spinner("Preparing answer..."):
                    qa_chain = get_qa_chain(vectorstore, final_selected_role, selected_language_code)
                    input_data = {"query": st.session_state.current_question_input}
                    response_data = qa_chain.invoke(input_data)
                    current_answer = response_data["result"]
                    current_sources = response_data.get("source_documents", [])
                    st.session_state.last_answer = current_answer
                    st.session_state.source_documents = current_sources
                    st.session_state.conversation_history.append({
                        "question": st.session_state.current_question_input, "answer": current_answer, "sources": current_sources,
                        "role": final_selected_role, "language": selected_language_label, "refined_answer": ""
                    })
            else:
                st.error("❌ Vector database could not be loaded. Please upload and process a PDF.")
                st.session_state.last_answer = ""
                st.session_state.source_documents = []

    with action_cols[1]:
        if st.button("🧮 Summarize", use_container_width=True):
            st.session_state.last_answer = ""
            st.session_state.refined_answer = ""
            st.session_state.source_documents = []
            st.session_state.concept_map_data = ""
            st.session_state.timeline_data = ""
            if all_chunks_for_session:
                with st.spinner("📚 Summarizing documents... This may take some time."):
                    summary = summarize_documents(all_chunks_for_session, final_selected_role, selected_language_code)
                    st.session_state.document_summary = summary
            else:
                st.warning("⚠️ No document found to summarize. Please upload a PDF first.")
                st.session_state.document_summary = ""

    with action_cols[2]:
        if st.button("🧠 Concept Map", use_container_width=True):
            st.session_state.last_answer = ""
            st.session_state.refined_answer = ""
            st.session_state.source_documents = []
            st.session_state.document_summary = ""
            st.session_state.timeline_data = ""
            if all_chunks_for_session:
                with st.spinner("🗺️ Creating concept map..."):
                    map_data = generate_concept_map_data(all_chunks_for_session, final_selected_role, selected_language_code)
                    st.session_state.concept_map_data = map_data
            else:
                st.warning("⚠️ No document found for concept map. Please upload a PDF first.")
                st.session_state.concept_map_data = ""

    with action_cols[3]:
        if st.button("⏳ Timeline", use_container_width=True):
            st.session_state.last_answer = ""
            st.session_state.refined_answer = ""
            st.session_state.source_documents = []
            st.session_state.document_summary = ""
            st.session_state.concept_map_data = ""
            if all_chunks_for_session:
                with st.spinner("📅 Extracting timeline..."):
                    timeline = extract_timeline_from_documents(all_chunks_for_session, final_selected_role, selected_language_code)
                    st.session_state.timeline_data = timeline
            else:
                st.warning("⚠️ No document found for timeline. Please upload a PDF first.")
                st.session_state.timeline_data = ""

    st.markdown("---")

    if st.session_state.document_summary:
        st.markdown("### 📜 Document Summary")
        st.write(st.session_state.document_summary)

    if st.session_state.concept_map_data:
        st.markdown("### 🗺️ Concept Map")
        if "```mermaid" in st.session_state.concept_map_data:
            st.markdown(st.session_state.concept_map_data)
        else:
            st.warning("Concept map could not be visualized. Raw data:")
            st.code(st.session_state.concept_map_data)

    if st.session_state.timeline_data:
        st.markdown("### 📅 Timeline")
        st.markdown(st.session_state.timeline_data)

    if st.session_state.last_answer:
        st.markdown("### 💡 Latest Answer")
        st.write(st.session_state.last_answer)

        col_refine1, col_refine2 = st.columns(2)
        with col_refine1:
            if st.button("🔁 Elaborate Answer"):
                from chatbot import refine_answer
                if st.session_state.last_question and st.session_state.last_answer:
                    with st.spinner("Elaborating answer..."):
                        refined_text = refine_answer(st.session_state.last_question, st.session_state.last_answer, "elaborate", final_selected_role, selected_language_code)
                        st.session_state.refined_answer = refined_text
                        if st.session_state.conversation_history:
                            st.session_state.conversation_history[-1]["refined_answer"] = refined_text
                else:
                    st.warning("You must have an answer first to elaborate.")

        with col_refine2:
            if st.button("🔀 Simplify Answer"):
                from chatbot import refine_answer
                if st.session_state.last_question and st.session_state.last_answer:
                    with st.spinner("Simplifying answer..."):
                        refined_text = refine_answer(st.session_state.last_question, st.session_state.last_answer, "simplify", final_selected_role, selected_language_code)
                        st.session_state.refined_answer = refined_text
                        if st.session_state.conversation_history:
                            st.session_state.conversation_history[-1]["refined_answer"] = refined_text
                else:
                    st.warning("You must have an answer first to simplify.")

        if st.session_state.refined_answer:
            st.markdown("#### ✨ Latest Refined Answer:")
            st.write(st.session_state.refined_answer)

        if st.session_state.source_documents:
            st.markdown("📚 **References:**")
            references = set()
            for doc in st.session_state.source_documents:
                source_name = doc.metadata.get("source", "Unknown Source")
                page_number = doc.metadata.get("page", "Unknown Page")
                references.add(f"- {source_name} (Page: {page_number})")
            for ref in sorted(list(references)):
                st.markdown(ref)

st.sidebar.title("📜 Conversation History")
if not st.session_state.conversation_history:
    st.sidebar.info("No conversation history yet.")
else:
    for i, entry in enumerate(reversed(st.session_state.conversation_history)):
        with st.sidebar.expander(f"Question {len(st.session_state.conversation_history) - i}: {entry['question'][:30]}..."):
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(f"**Role:** {entry['role']}")
            st.markdown(f"**Language:** {entry['language']}")
            st.markdown("**Answer:**")
            st.write(entry['answer'])
            if entry.get('refined_answer'):
                st.markdown("**Refined Answer:**")
                st.write(entry['refined_answer'])
            if entry['sources']:
                st.markdown("**References:**")
                current_references = set()
                for doc_ref in entry['sources']:
                    source_name_ref = doc_ref.metadata.get("source", "Unknown Source")
                    page_number_ref = doc_ref.metadata.get("page", "Unknown Page")
                    current_references.add(f"- {source_name_ref} (Page: {page_number_ref})")
                for r_ref in sorted(list(current_references)):
                    st.markdown(r_ref)
