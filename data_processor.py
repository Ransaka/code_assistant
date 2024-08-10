import os
from typing import List
import numpy as np
import redis
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
import time

from redis.commands.search.field import (
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sourcegraph import Sourcegraph

load_dotenv()

INDEX_NAME = "idx:codes_vss"

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

def fetch_data(url):
    def get_description(code):
      chat_session = model.start_chat(
        history=[
          {
            "role": "user",
            "parts": [
              f"Code: {code}",
            ],
          },
        ]
      )
      response = chat_session.send_message("INSERT_INPUT_HERE")

      return response.text
    gihub_repository = Sourcegraph(url)
    gihub_repository.run()
    data = dict(gihub_repository.node_data)
    for key, value in tqdm(data.items()):
      data[key]['description'] = get_description(value['definition'])
      data[key]['uses'] = ", ".join(list(gihub_repository.get_dependencies(key)))
      time.sleep(2) #to overcome limit issues
    return data

def get_embeddings(content: List):
    return genai.embed_content(model='models/text-embedding-004',content=content)['embedding']

def ingest_data(client: redis.Redis, data):
    try:
       client.delete(client.keys("code:*"))
    except:
       pass
    pipeline = client.pipeline()
    for i, code_metadata in enumerate(data.values(), start=1):
        redis_key = f"code:{i:03}"
        pipeline.json().set(redis_key, "$", code_metadata)
    _ = pipeline.execute()
    keys = sorted(client.keys("code:*"))
    defs = client.json().mget(keys, "$.definition")
    descs = client.json().mget(keys, "$.description")
    embed_inputs = []

    for i in range(1, len(keys)+1):
        embed_inputs.append(
            f"""{defs[i-1][0]}\n\n{descs[i-1][0]}"""
        )
    embeddings = get_embeddings(embed_inputs)
    VECTOR_DIMENSION = len(embeddings[0])
    pipeline = client.pipeline()
    for key, embedding in zip(keys, embeddings):
        pipeline.json().set(key, "$.embeddings", embedding)
    pipeline.execute()

    schema = (
        TextField("$.name", no_stem=True, as_name="name"),
        TagField("$.type", as_name="type"),
        TextField("$.definition", no_stem=True, as_name="definition"),
        TextField("$.file_name", no_stem=True, as_name="file_name"),
        TextField("$.description", no_stem=True, as_name="description"),
        TextField("$.uses", no_stem=True, as_name="uses"),
        VectorField(
            "$.embeddings",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIMENSION,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="vector",
        ),
    )
    definition = IndexDefinition(prefix=["code:"], index_type=IndexType.JSON)
    try:
       _ = client.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
    except redis.exceptions.ResponseError:
       client.ft(INDEX_NAME).dropindex()
       _ = client.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
    info = client.ft(INDEX_NAME).info()
    num_docs = info["num_docs"]
    indexing_failures = info["hash_indexing_failures"]
    return f"{num_docs} documents indexed with {indexing_failures} failures"