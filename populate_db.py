import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import weaviate
from weaviate.embedded import EmbeddedOptions

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Ensure the 'OPENAI_API_KEY' environment variable is set.")

client = weaviate.Client(
    embedded_options=EmbeddedOptions(
        additional_env_vars={
        "ENABLE_MODULES":
        "text2vec-openai"}
    ),
    additional_headers={
        'X-OpenAI-Api-Key': api_key
    }
)

client.schema.delete_all()
client.schema.get()

def combine_csv_files(directory):
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]

    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities if any
    combined_df.fillna(0, inplace=True)
    return combined_df


# Specify the directory containing your CSV files
directory = './source_data'
df = combine_csv_files(directory)

print(df.head())

client.schema.create_class({
    "class": "Transcript",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text"
        }
    },
    "properties": [
        {"name": "text", "dataType": ["text"], "indexInverted": True},
        {"name": "speaker", "dataType": ["string"]},
        {"name": "timestamp", "dataType": ["string"]}
    ],
})

def add_to_weaviate(text, speaker, timestamp):
    try:
        client.data_object.create(
            data_object={
                "text": text,
                "speaker": speaker,
                "timestamp": timestamp
            },
            class_name="Transcript"
        )
    except Exception as e:
        print(f"Error in adding data to Weaviate: {str(e)}")

def validate_data(row):
    # Example validation logic
    if pd.isna(row['Text']) or pd.isna(row['Speaker']) or pd.isna(row['Timestamp']):
        return False
    return True

for index, row in df.iterrows():
    if validate_data(row):
        add_to_weaviate(row['Text'], row.get('Speaker', ""), row.get('Timestamp', ""))
    else:
        print(f"Invalid data at index {index}: {row}")

result = (
    client.query.aggregate("Transcript")
    .with_fields("meta { count }")
    .do()
)
print("Object count: ", result["data"]["Aggregate"]["Transcript"], "\n")
