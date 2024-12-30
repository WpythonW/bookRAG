import streamlit as st
from jinja2 import Template
import sys
from typing import Dict, Any, Optional, List
import json
import io
from contextlib import contextmanager
from gemini_llm import GeminiLLM
from chroma_client import ChromaDBManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from book_summarizer import TextProcessor
import logging

logger = logging.getLogger(__name__)

### Цитирование только по реальным фрагментам! История для конеткста
PROMPT_TEMPLATE = """
{# Main sections #}
# Scene summaries
{{ story }}

# Relevant text excerpts
{{ search_results }}

{# Instructions for using sections #}
When answering questions:
1. Use "Search results" for:
  - Direct character quotes
  - Specific event/situation details
  - Exact dialogue details 
  - Text fact verification

2. Use "Scene summaries" only for:
  - Understanding overall context
  - Clarifying event timeline
  - Connecting different text parts

{# User question #}
Question: {{ user_question }}
"""

class PromptManager:
    def __init__(self, chroma_manager, gemini_client):
        self.chroma_manager = chroma_manager
        self.gemini_client = gemini_client
        self.template = Template(PROMPT_TEMPLATE)

    def create_prompt(self, collection_name: str, query: str) -> Dict[str, Any]:
        story = self.chroma_manager.get_complete_story(collection_name)
        search_results = self.chroma_manager.search(collection_name, query)
        
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            scene = result['scene']
            formatted_scene = (
                f"\n## Info for scene:\n"
                f"Description: {scene['document'].replace('search_document:', '').strip()}\n"
                f"Location: {scene['metadata']['location']}\n"
                f"Characters: {scene['metadata']['who']}\n"
                "### Relevant text fragments\n")
            
            for j, chunk in enumerate(result['chunks'], 1):
                formatted_scene += f"#### Fragment {j}\n{chunk['document']}\n"
                
            formatted_results.append(formatted_scene)
            
        formatted_search_results = "\n".join(formatted_results)
        
        prompt = self.template.render(
            story=story,
            search_results=formatted_search_results,
            user_question=query
        )
        return {
            "prompt": prompt
        }

def clear_chat_history():
    st.session_state.messages = []

def process_user_message(prompt: str, collection_name: str, prompt_manager: PromptManager) -> str:
    if not collection_name:
        return prompt
        
    prompt_data = prompt_manager.create_prompt(
        collection_name=collection_name,
        query=prompt
    )
    return prompt_data["prompt"]

st.set_page_config(
    page_title="Gemini Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 450px;
            max-width: 450px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Gemini Chat")

if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "file_processing_complete" not in st.session_state:
    st.session_state.file_processing_complete = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

@st.cache_resource
def get_chroma_manager():
    return ChromaDBManager()

@st.cache_resource
def get_llm():
    return GeminiLLM()

@st.cache_resource
def get_prompt_manager():
    manager = get_chroma_manager()
    llm = get_llm()
    return PromptManager(manager, llm)

manager = get_chroma_manager()
llm = get_llm()

@contextmanager
def capture_output():
    new_out = io.StringIO()
    old_out = sys.stdout
    sys.stdout = new_out
    try:
        yield new_out
    finally:
        sys.stdout = old_out

def process_file(file_content, file_name):
    try:
        base_name = file_name.rsplit('.', 1)[0]
        
        if manager.has_collections(base_name):
            st.toast(f"Коллекция {base_name} уже существует", icon="ℹ️")
            st.session_state.processed_files.add(file_name)
            return True

        chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=19000,
            chunk_overlap=0,
            length_function=len,
            separators=['\n', '\n\n', '\n\n\n', '\n\n\n\n']
        )
        
        book_chunks = chunk_splitter.split_text(file_content)
        
        llm = GeminiLLM(temperature=0.1)
        processor = TextProcessor(llm)
        
        with st.sidebar:
            with st.spinner("Обработка текста..."):
                summaries = processor.process_book(book_chunks)
                
                try:
                    scenes_collection, chunks_collection = manager.create_collections(
                        collection_name=base_name,
                        scenes=summaries.all_scenes,
                        book_chunks=book_chunks
                    )
                    st.toast(f"Обработка файла {file_name} завершена успешно", icon="✅")
                    st.session_state.processed_files.add(file_name)
                    return True
                except Exception as e:
                    st.error(f"Ошибка при создании коллекций: {str(e)}")
                    return False
                    
    except Exception as e:
        st.error(f"Ошибка при обработке: {str(e)}")
        return False

with st.sidebar:
    st.header("Настройки")
    
    st.subheader("Существующие коллекции")
    
    collections = manager.get_base_collection_names()
    
    if collections:
        selected_collection = st.selectbox(
            "Выберите книгу",
            options=collections,
            index=collections.index(st.session_state.active_collection) if st.session_state.active_collection in collections else 0
        )
        
        if selected_collection:
            st.session_state.active_collection = selected_collection
            st.success(f"Активная книга: {st.session_state.active_collection}")
            
            # Размещаем кнопки в две колонки одинакового размера
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑 Удалить книгу\n(саммари и векторы)", use_container_width=True):
                    manager.delete_collections(selected_collection)
                    st.session_state.active_collection = None
                    st.rerun()

            with col2:
                if st.button("🔄 Очистить историю чата", use_container_width=True):
                    clear_chat_history()
                    st.rerun()
    else:
        st.info("Нет существующих коллекций")
    
    st.divider()
    
    st.header("Загрузка файлов")
    uploaded_files = st.file_uploader(
        "Выберите файлы",
        accept_multiple_files=True,
        type=['txt'],
        key=f"file_uploader_{st.session_state.uploader_key}"
    )
    
    if uploaded_files and not st.session_state.file_processing_complete:
        files_to_process = [file for file in uploaded_files 
                          if file.name not in st.session_state.processed_files]
        
        if files_to_process:
            for file in files_to_process:
                file_content = file.read()
                try:
                    text_content = file_content.decode('utf-8')
                    process_file(text_content, file.name)
                except Exception as e:
                    st.error(f"Ошибка при обработке: {str(e)}")
                file.seek(0)
            
            if all(file.name in st.session_state.processed_files for file in uploaded_files):
                st.toast("Все файлы успешно обработаны", icon="🎉")
                st.session_state.file_processing_complete = True
                st.session_state.processed_files.clear()
                st.session_state.uploader_key += 1
                st.rerun()
    elif not uploaded_files:
        st.session_state.file_processing_complete = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["parts"][0]["text"])

if prompt := st.chat_input("Введите ваше сообщение..."):
    st.session_state.messages.append({"role": "user", "parts": [{"text": prompt}]})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with capture_output() as output:
            try:
                prompt_manager = get_prompt_manager()
                
                full_prompt = process_user_message(
                    prompt=prompt,
                    collection_name=st.session_state.active_collection,
                    prompt_manager=prompt_manager
                )
                
                response = llm.generate_response(
                    full_prompt,
                    chat_history=st.session_state.messages[:-1]
                )
                logger.info(f"Token stats: {response['tokens']}")

                message_placeholder.markdown(response["text"])
                
                with st.expander("Показать промпт"):
                    st.code(full_prompt, language="text")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "parts": [{"text": response["text"]}]
                })
                
            except Exception as e:
                error_message = f"Ошибка: {str(e)}"
                message_placeholder.error(error_message)
                st.error(error_message)
            
            console_output = output.getvalue().strip()
            if console_output:
                st.info(console_output)


