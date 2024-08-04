import os
import pandas as pd
from dotenv import load_dotenv
from itertools import islice
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import time

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

# Define the model name and create an embedding instance
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name, openai_api_key=os.getenv('OPENAI_API_KEY'))

# Create a new Pinecone index if it doesn't exist
index_name = "food-advisor"
if index_name not in pc.list_indexes():
    spec = ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust the dimension based on your embedding model
        metric="cosine",
        spec=spec
    )

# Connect to the Pinecone index
index = pc.Index(index_name)
vectordb = LangchainPinecone(index, embed.embed_query, "text")

# Load the dataset
df = pd.read_csv('recipes_final.csv')
df = df.fillna('')

# Function to convert a tuple to a Document object
def tuple_to_document(t: tuple):
    return Document(
        page_content=t.Description,
        metadata={
            'name': t.Name,
            'time': t.TotalTime,
            'category': t.RecipeCategory,
            'keywords': t.Keywords,
            'ingredients': t.Ingredients,
            'calories': t.Calories,
            'carbohydrates percentage': t.CarbohydratePercentage,
            'proteins percentage': t.ProteinPercentage,
            'fat percentage': t.FatPercentage,
            'sugar percentage': t.SugarPercentage,
            'instructions': t.RecipeInstructions,
            'yields': t.RecipeYield
        }
    )

# Convert DataFrame rows to Document objects
docs = [tuple_to_document(row) for row in tqdm(df.itertuples(index=False))]

# Function to split a list into chunks
def chunks(lst, chk_size):
    lst_it = iter(lst)
    return iter(lambda: tuple(islice(lst_it, chk_size)), ())

# Split documents into chunks
chunkSize = 100
docs_chunked_list = list(chunks(docs, chunkSize))

# Upload documents to the vector database
print('Uploading to vector db')
s = time.perf_counter()
for docs_chunk in docs_chunked_list:
    vectordb.add_documents(docs_chunk)
elapsed = time.perf_counter() - s
print("\033[1m" + f"Upload executed in {elapsed:0.2f} seconds." + "\033[0m")