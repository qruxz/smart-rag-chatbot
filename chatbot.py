from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from prompts import get_prompt_template

def get_qa_chain(vectorstore, role, language_code="tr"):
    llm = OllamaLLM(model="gemma:2b")
    prompt = get_prompt_template(role, language_code)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

def refine_answer(original_question, original_answer, refinement_type, role, language_code="tr"):
    llm = OllamaLLM(model="gemma:2b")
    if refinement_type == "detaylandır":
        refinement_instruction = "Expand the above answer with more technical terms and explanations."
    elif refinement_type == "sadeleştir":
        refinement_instruction = "Rephrase the above answer in simpler language that anyone can understand."
    else:
        return "Invalid refinement type."
    language_instruction = ""
    if language_code == "en":
        language_instruction = "Provide the refined answer in English."
    elif language_code == "tr":
        language_instruction = "Provide the refined answer in Turkish."
    prompt_text = f"""You are acting as a '{role}'.
User's question: "{original_question}"
Initial answer: "{original_answer}"

TASK: {refinement_instruction}
{language_instruction}
Only provide the refined answer.
"""
    try:
        refined_response = llm.invoke(prompt_text)
        return refined_response 
    except Exception as e:
        print(f"Error while refining answer: {e}")
        return "An error occurred while refining the answer."

def generate_suggested_questions(document_chunks, role, language_code="tr", num_questions=3):
    llm = OllamaLLM(model="gemma:2b")
    context_text = ""
    char_limit = 2000
    for chunk_doc in document_chunks:
        if len(context_text) + len(chunk_doc.page_content) < char_limit:
            context_text += chunk_doc.page_content + "\n\n"
        else:
            break
    if not context_text:
        context_text = "General questions about the document content."
    if language_code == "en":
        question_language_instruction = f"Generate {num_questions} insightful questions about the following content, in English. The user will be interacting as a '{role}'."
        output_format_instruction = "Provide only the questions, each on a new line. Do not number them or add any other text."
    elif language_code == "tr":
        question_language_instruction = f"Generate {num_questions} thought-provoking questions in Turkish that a user in the role of '{role}' might ask about the content below."
        output_format_instruction = "Only provide the questions, each on a new line. Do not number or add any extra text."
    prompt_text = f"""{question_language_instruction}

Content Summary:
---
{context_text.strip()}
---

{output_format_instruction}
"""
    try:
        response = llm.invoke(prompt_text)
        suggested_questions = [q.strip() for q in response.split('\n') if q.strip()]
        return suggested_questions[:num_questions]
    except Exception as e:
        print(f"Error while generating suggested questions: {e}")
        return []

def summarize_documents(document_chunks, role, language_code="tr"):
    llm = OllamaLLM(model="gemma:2b")
    full_text = ""
    char_limit_for_summary = 10000
    for chunk_doc in document_chunks:
        if len(full_text) + len(chunk_doc.page_content) < char_limit_for_summary:
            full_text += chunk_doc.page_content + "\n\n"
        else:
            full_text += "\n\n[Part of the content was truncated due to token limits.]"
            break
    if not full_text.strip():
        return "No content available to summarize."
    if language_code == "en":
        summary_language_instruction = "Provide the summary in English."
        role_instruction = f"You are acting as a '{role}'."
        task_instruction = "Generate a comprehensive summary of the following text."
    else:
        summary_language_instruction = "Provide the summary in Turkish."
        role_instruction = f"You are acting as a '{role}'."
        task_instruction = "Generate a comprehensive summary of the following text."
    prompt_text = f"""{role_instruction}
{task_instruction}
{summary_language_instruction}

Text:
---
{full_text.strip()}
---

Please provide a well-structured summary including the main points of the above text.
"""
    try:
        response = llm.invoke(prompt_text)
        return response
    except Exception as e:
        print(f"Error while generating summary: {e}")
        return "An error occurred while generating the summary."

def extract_keywords_from_documents(document_chunks, role, language_code="tr", num_keywords=10):
    llm = OllamaLLM(model="gemma:2b")
    full_text = ""
    char_limit_for_keywords = 5000 
    for chunk_doc in document_chunks:
        if len(full_text) + len(chunk_doc.page_content) < char_limit_for_keywords:
            full_text += chunk_doc.page_content + "\n\n"
        else:
            break
    if not full_text.strip():
        return []
    if language_code == "en":
        role_instruction = f"You are acting as a '{role}'."
        task_instruction = f"Extract the top {num_keywords} keywords or concepts from the following text."
        output_format_instruction = "List the keywords as a single comma-separated string. Add nothing else."
        language_preference = "English"
    else:
        role_instruction = f"You are acting as a '{role}'."
        task_instruction = f"Extract the top {num_keywords} keywords or concepts from the following text."
        output_format_instruction = "List the keywords as a single comma-separated string. Add nothing else."
        language_preference = "Turkish"
    prompt_text = f"""{role_instruction}
{task_instruction}
Language Preference: {language_preference}.

Text:
---
{full_text.strip()}
---

{output_format_instruction}
"""
    try:
        response = llm.invoke(prompt_text)
        keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
        return keywords[:num_keywords]
    except Exception as e:
        print(f"Error while extracting keywords: {e}")
        return []

def generate_concept_map_data(document_chunks, role, language_code="tr"):
    llm = OllamaLLM(model="gemma:2b")
    full_text = ""
    char_limit_for_map = 7000
    for chunk_doc in document_chunks:
        if len(full_text) + len(chunk_doc.page_content) < char_limit_for_map:
            full_text += chunk_doc.page_content + "\n\n"
        else:
            break
    if not full_text.strip():
        return "No content found for concept map."
    if language_code == "en":
        role_instruction = f"As a '{role}', analyze the main concepts and their relationships in the following text."
        task_instruction = "Based on this analysis, create a text-based concept map. Provide the output in Mermaid.js 'graph TD' or 'graph LR' format. The map should hierarchically show the main ideas and their sub-topics."
        language_preference = "English"
    else:
        role_instruction = f"As a '{role}', analyze the main concepts and their relationships in the following text."
        task_instruction = "Based on this analysis, create a text-based concept map. Provide the output in Mermaid.js 'graph TD' or 'graph LR' format. The map should hierarchically show the main ideas and their sub-topics."
        language_preference = "Turkish"
    output_example = """
Example Output Format (Mermaid.js graph TD):
```mermaid
graph TD
    A[Main Concept] --> B(Sub Concept 1)
    A --> C(Sub Concept 2)
    B --> D{Detail 1.1}
    B --> E{Detail 1.2}
    C --> F{Detail 2.1}
```
"""

    language_preference = "Türkçe"

    if language_code == "en":
        role_instruction = f"As a '{role}', analyze the main concepts and their relationships in the following text."
        task_instruction = "Based on this analysis, create a text-based concept map. Provide the output in Mermaid.js 'graph TD' or 'graph LR' format. The map should hierarchically show the main ideas and their sub-topics."
        language_preference = "English (for node labels if possible, but structure is key)"
    
    prompt_text = f"""{role_instruction}
{task_instruction}
Dil Tercihi (kavram etiketleri için mümkünse): {language_preference}.

Metin:
---
{full_text.strip()}
---

{output_example}
Lütfen SADECE Mermaid.js kod bloğunu (` ```mermaid ... ``` `) yanıt olarak ver. Başka hiçbir açıklama veya metin ekleme.
"""

    try:
        response = llm.invoke(prompt_text)
        # Modelin doğrudan ```mermaid ... ``` bloğunu döndürdüğünü varsayıyoruz.
        # Eğer değilse, bu bloğu ayıklamak için ek işlem gerekebilir.
        if "```mermaid" in response and "```" in response.split("```mermaid")[1]:
            mermaid_code = "```mermaid" + response.split("```mermaid")[1].split("```")[0] + "```"
            return mermaid_code.strip()
        else: # Basit bir fallback veya hata
            print(f"Model beklenen Mermaid formatında yanıt vermedi: {response}")
            # Belki de sadece metin tabanlı bir hiyerarşi istemek daha güvenli olabilir.
            # Şimdilik bu şekilde bırakalım.
            return "Konsept haritası üretilemedi (beklenen formatta değil)." 
            
    except Exception as e:
        print(f"Konsept haritası üretilirken hata: {e}")
        return "Konsept haritası üretilirken bir sorun oluştu."

def extract_timeline_from_documents(document_chunks, role, language_code="tr"):
    """
    Yüklenen belgelerden tarihsel olayları çıkarıp bir zaman çizelgesi oluşturur.
    """
    llm = OllamaLLM(model="gemma:2b")

    # Metnin tamamını veya önemli bir kısmını alalım
    full_text = ""
    char_limit_for_timeline = 8000 # Zaman çizelgesi için daha fazla bağlam gerekebilir
    for chunk_doc in document_chunks:
        if len(full_text) + len(chunk_doc.page_content) < char_limit_for_timeline:
            full_text += chunk_doc.page_content + "\n\n"
        else:
            break
    
    if not full_text.strip():
        return "Zaman çizelgesi için içerik bulunamadı."

    # Dil ve rol talimatları
    role_instruction = f"Bir '{role}' olarak, aşağıdaki metindeki tarihleri ve bu tarihlerle ilişkili önemli olayları veya bilgileri analiz et."
    task_instruction = "Bu analize dayanarak, kronolojik olarak sıralanmış bir zaman çizelgesi oluştur. Her bir maddeyi 'Tarih: Açıklama' formatında listele."
    language_preference = "Türkçe"

    if language_code == "en":
        role_instruction = f"As a '{role}', analyze the following text to identify dates and the significant events or information associated with them."
        task_instruction = "Based on this analysis, create a chronologically ordered timeline. List each item in the format 'Date: Description'."
        language_preference = "English"
    
    prompt_text = f"""{role_instruction}
{task_instruction}
Dil Tercihi: {language_preference}.

Metin:
---
{full_text.strip()}
---

Lütfen bulunan olayları kronolojik sıraya göre, her biri yeni bir satırda olacak şekilde listele. Eğer metinde belirgin tarihler yoksa, "Belgede belirgin bir zaman çizelgesi bulunamadı." yanıtını ver.
"""

    try:
        response = llm.invoke(prompt_text)
        # Yanıtın doğrudan markdown listesi veya ilgili mesaj olduğunu varsayıyoruz.
        return response 
    except Exception as e:
        print(f"Zaman çizelgesi çıkarılırken hata: {e}")
        return "Zaman çizelgesi çıkarılırken bir sorun oluştu."