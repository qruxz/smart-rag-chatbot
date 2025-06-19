from langchain.prompts import PromptTemplate

def get_prompt_template(role_param: str, language_code: str = "tr"):
    language_instruction = ""
    if language_code == "en":
        language_instruction = "Please provide the answer in English."
    elif language_code == "tr":
        language_instruction = "Please provide the answer in Turkish."

    template_string = f"""
You are an artificial intelligence acting like a {role_param}.
The user has uploaded the following PDF document:

{{context}}

User's question:
{{question}}

Please answer this question like a {role_param}, in a detailed and understandable way.
{language_instruction}
Don't forget to reference the content of the document when answering.
"""
    return PromptTemplate(input_variables=["context", "question"], template=template_string)
