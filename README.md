# Mr. Research

An application that helps find relevant answers in transcripts.

## Installation

1. Clone repo locally
1. Install dependencies

```
pip install weaviate-client
pip install openai
pip install dotenv
```

1. Create a `.env` file with the proper key/value pairs:

```
OPENAI_API_KEY=
PERSISTENCE_DATA_PATH='./weaviate_data'
LOG_LEVEL=error
```

## Usage

### Generate transcript DB

1. Ensure source files are in `./source_data` and in the right format
2. Run "populate_db.py"

### Query DB

1. Run "query.py"
2. Ask questions directly in terminal
# mnemosyne
