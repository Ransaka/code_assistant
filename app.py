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

load_dotenv()

# Initialize Streamlit app
st.title("Code Assistant Chat")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

client = redis.Redis(
  host=os.environ['REDIS_HOST'],
  port=12305,
  password=os.environ['REDIS_PASSWORD'])

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


def get_related_functions(query: str) -> str:
    """
    Perform a vector similarity search and retrieve related functions.

    Args:
        query (str): The input query to encode.

    Returns:
        str: A formatted string containing details of related functions.
    """
    INDEX_NAME = "idx:codes_vss"
    vector_search_query = (
            Query('(*)=>[KNN 3 @vector $query_vector AS vector_score]')
            .sort_by('vector_score')
            .return_fields('vector_score', 'id', 'name', 'definition', 'file_name', 'type', 'uses')
            .dialect(2)
        )
    
    encoded_query = get_embeddings(query)
    vector_params = {
        "query_vector": np.array(encoded_query, dtype=np.float32).tobytes()
    }
    
    # Perform initial vector search
    result_docs = client.ft(INDEX_NAME).search(vector_search_query, vector_params).docs
    
    # Extract related function names and uses
    related_items: List[str] = []
    for doc in result_docs:
        related_items.append(doc.name)
        related_items.extend(use for use in doc.uses.split(", ") if use)
    
    # Construct and execute secondary search query
    secondary_query = Query(f"@name:({' | '.join(set(related_items))})").return_fields(
        'id', 'name', 'definition', 'file_name', 'type'
    )
    docs = client.ft(INDEX_NAME).search(secondary_query).docs
    
    # Format results
    formatted_results = []
    for doc in docs:
        formatted_results.append(
            f"User Question: {query}\n\n"
            f"{'*' * 28} CODE SNIPPET {doc.id} {'*' * 28}\n"
            f"* Name: {doc.name}\n"
            f"* File: {doc.file_name}\n"
            f"* {doc.type.capitalize()} definition:\n"
            f"```python\n{doc.definition}\n```\n"
        )
    
    return "\n\n".join(formatted_results)

@component
class RedisRetriever:
  @component.output_types(context=str)
  def run(self, query:str):
    return {"context": get_related_functions(query)}

template = """
You are a helpfull agent optimized for finding issues in codes and give most accurate answers / sugestions to the user. You have provided with all nessasary codes for fixing this issue.
First you should understand user question and potential code block to look. then craft reply carefully and on-point as it allow user to find exact solution for their issue. Give codes not just openions.
However you are free to perform this task list outside of above:
 * greet the user
 * [ONLY IF QUESTION IS NOT SUFFICIENT] ask about more clarity
 * Politly reject un-wanted user queries

{{context}}
"""

prompt_builder = PromptBuilder(template=template)

pipeline = Pipeline()
pipeline.add_component(name="Retriever", instance=RedisRetriever())
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", gemini)
pipeline.connect("Retriever.context", "prompt_builder")
pipeline.connect("prompt_builder", "llm")
pipeline.draw(path='pipeline.png')

# question = "Tokenization train operation takes a lot of time to execute in larger datasets"

# response = pipeline.run({"Retreiver": {"query": question}})

# llm_response = response["llm"]["replies"][0]


# Chat input
if prompt := st.chat_input("What's your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")
        
        try:
            response = pipeline.run({"Retriever": {"query": prompt}})
            llm_response = response["llm"]["replies"][0]
            
            response_placeholder.markdown(llm_response)
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
        except Exception as e:
            response_placeholder.markdown(f"An error occurred: {str(e)}")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()