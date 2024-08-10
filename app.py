import redis 
import os 
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List
import numpy as np 
from redis.commands.search.query import Query
from haystack import Pipeline, component
from haystack.utils import Secret
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator, GoogleAIGeminiGenerator
from haystack.components.builders import PromptBuilder
import streamlit as st
from data_processor import fetch_data,ingest_data

load_dotenv()


genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="You are optimized to generate accurate descriptions for given Python codes. When the user inputs the code, you must return the description according to its goal and functionality.  You are not allowed to generate additional details. The user expects at least 5 sentence-long descriptions.",
)

gemini = GoogleAIGeminiGenerator(api_key=Secret.from_env_var("GEMINI_API_KEY"), model='gemini-1.5-flash')

def get_embeddings(content: List):
    return genai.embed_content(model='models/text-embedding-004',content=content)['embedding']


def draft_prompt(query: str, chat_history: str) -> str:
    """
    Perform a vector similarity search and retrieve related functions.

    Args:
        query (str): The input query to encode.

    Returns:
        str: A formatted string containing details of related functions.
    """
    INDEX_NAME = "idx:codes_vss"
    client = st.session_state.client
    vector_search_query = (
        Query('(*)=>[KNN 2 @vector $query_vector AS vector_score]')
        .sort_by('vector_score')
        .return_fields('vector_score', 'id', 'name', 'definition', 'file_name', 'type', 'uses')
        .dialect(2)
    )
    
    encoded_query = get_embeddings(query)
    vector_params = {
        "query_vector": np.array(encoded_query, dtype=np.float32).tobytes()
    }
    
    result_docs = client.ft(INDEX_NAME).search(vector_search_query, vector_params).docs
    
    related_items: List[str] = []
    dependencies: List[str] = []
    for doc in result_docs:
        related_items.append(doc.name)
        if doc.uses:
            dependencies.extend(use for use in doc.uses.split(", ") if use)
    
    dependencies = list(set(dependencies) - set(related_items))
    
    def get_query(item_list):
        return Query(f"@name:({' | '.join(item_list)})").return_fields(
            'id', 'name', 'definition', 'file_name', 'type'
        )
    
    related_docs = client.ft(INDEX_NAME).search(get_query(related_items)).docs
    dependency_docs = client.ft(INDEX_NAME).search(get_query(dependencies)).docs
    
    def format_doc(doc):
        return (
            f"{'*' * 28} CODE SNIPPET {doc.id} {'*' * 28}\n"
            f"* Name: {doc.name}\n"
            f"* File: {doc.file_name}\n"
            f"* {doc.type.capitalize()} definition:\n"
            f"```python\n{doc.definition}\n```\n"
        )
    
    formatted_results_main = [format_doc(doc) for doc in related_docs]
    formatted_results_support = [format_doc(doc) for doc in dependency_docs]
    
    return (
        f"User Question: {query}\n\n"
        f"Current Chat History: \n{chat_history}\n\n"
        f"USE BELOW CODES TO ANSWER USER QUESTIONS.\n"
        f"{chr(10).join(formatted_results_main)}\n\n"
        f"SOME SUPPORTING FUNCTIONS AND CLASS YOU MAY WANT.\n"
        f"{chr(10).join(formatted_results_support)}"
    )

@component
class RedisRetreiver:
  @component.output_types(context=str)
  def run(self, query:str, chat_history:str):
    return {"context": draft_prompt(query, chat_history)}

llm = GoogleAIGeminiGenerator(api_key=Secret.from_env_var("GEMINI_API_KEY"), model='gemini-1.5-pro')
# llm = OpenAIGenerator()

template = """
You are a helpful agent optimized to resolve GitHub issues for your organization's libraries. Users will ask questions when they encounter problems with the code repository.
You have access to all the necessary code for addressing these issues. 
First, you should understand the user's question and identify the relevant code blocks. 
Then, craft a precise and targeted response that allows the user to find an exact solution to their problem. 
You must provide code snippets rather than just opinions.
You should always assume user has installed this python package in their system and raised question raised while they are using the library.

In addition to the above tasks, you are free to:
 * Greet the user.
 * [ONLY IF THE QUESTION IS INSUFFICIENT] Request additional clarity.
 * Politely decline irrelevant queries.
 * Inform the user if their query cannot be processed or accomplished.

By any chance you should NOT,
 * Ask or recommend user to use different library. Or code snipits related to other similar libraies.
 * Provide inaccurate explnations.
 * Provide sugestions without code examples.

{{context}}
"""

prompt_builder = PromptBuilder(template=template)

pipeline = Pipeline()
pipeline.add_component(name="retriever", instance=RedisRetreiver())
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm)
pipeline.connect("retriever.context", "prompt_builder")
pipeline.connect("prompt_builder", "llm")

# Initialize Streamlit app
st.title("Code Assistant Chat")

tabs = ["Data Fetching","Assistant"]
selected_tab = st.sidebar.radio("Select a Tab", tabs)
if selected_tab == 'Data Fetching':
    if 'redis_connected' not in st.session_state:
        st.session_state.redis_connected = False

    if not st.session_state.redis_connected:
        st.header("Redis Connection Settings")
        
        redis_host = st.text_input("Redis Host")
        redis_port = st.number_input("Redis Port", min_value=1, max_value=65535, value=6379)
        redis_password = st.text_input("Redis Password", type="password")
        
        if st.button("Connect to Redis"):
            try:
                client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password
                )
                
                if client.ping():
                    st.success("Successfully connected to Redis!")
                    st.session_state.redis_connected = True
                    st.session_state.client = client
                else:
                    st.error("Failed to connect to Redis. Please check your settings.")
            except redis.ConnectionError:
                st.error("Failed to connect to Redis. Please check your settings and try again.")
    
    if st.session_state.redis_connected:
        url = st.text_input("Enter git clone URL")
        if url:
            with st.spinner("Fetching data..."):
                data = fetch_data(url)
            
            with st.spinner("Ingesting data..."):
                response_string = ingest_data(st.session_state.client, data)
            
            st.write(response_string)
if selected_tab == 'Assistant':
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.session_state.response = None

    if prompt := st.chat_input("What's your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")
            
            try:
                response = pipeline.run({"retriever": {"query": prompt, "chat_history": st.session_state.messages}}, include_outputs_from=['prompt_builder'])
                st.session_state.response = response
                llm_response = response["llm"]["replies"][0]
                
                response_placeholder.markdown(llm_response)
                st.session_state.messages.append({"role": "assistant", "content": llm_response})
            except Exception as e:
                response_placeholder.markdown(f"An error occurred: {str(e)}")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

    with st.expander("See Chat History"):
        st.markdown(st.session_state.response)