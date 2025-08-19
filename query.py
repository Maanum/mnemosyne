import os
import weaviate
import json
import logging
from dotenv import load_dotenv
from weaviate.embedded import EmbeddedOptions
from openai import OpenAI

logging.basicConfig(level=logging.ERROR)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "API key not found. Ensure the 'OPENAI_API_KEY' environment variable is set."
    )

clientApi = OpenAI(
    api_key=api_key,
    organization="",
    project="",
)


# Initialize the Weaviate client
clientDb = weaviate.Client(
    embedded_options=EmbeddedOptions(
        additional_env_vars={"ENABLE_MODULES": "text2vec-openai"}
    ),
    additional_headers={"X-OpenAI-Api-Key": api_key},
)


def load_llm_prompt():
    with open("llm_prompt.json", "r") as file:
        messages = json.load(file)
    return messages


prompt = load_llm_prompt()


def search_weaviate(query):
    try:

        excluded_speakers = []

        result = (
            clientDb.query.get(
                "Transcript", ["text", "speaker", "timestamp"]
            ).with_near_text({"concepts": [query]})
            # .with_limit(100)
            .do()
        )
        transcripts = result["data"]["Get"]["Transcript"]

        # Filter out transcripts where the speaker is 'Kristofer'
        filtered_transcripts = [
            transcript
            for transcript in transcripts
            if transcript.get("speaker") not in excluded_speakers
        ]
        # print(filtered_transcripts)
        return filtered_transcripts  # Ensure this returns a list of dicts
    except Exception as e:
        print(f"Error querying Weaviate: {str(e)}")
        return []


def format_for_openai(transcripts):
    context_parts = []
    for transcript in transcripts:
        speaker = transcript.get("speaker", "Unknown")
        timestamp = transcript.get("timestamp", "Unknown Time")
        text = transcript.get("text", "No text available")
        context_part = f"{text} ({speaker}, {timestamp})"
        context_parts.append(context_part)
    context = " ".join(context_parts)
    return context


def ask_openai(question, context):
    try:
        messages = [
            prompt,
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]
        completion = clientApi.chat.completions.create(model="gpt-4", messages=messages)
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error querying OpenAI: {str(e)}")
        return "I'm sorry, I couldn't generate a response."


def truncate_context(context, max_length=2048):
    # Tokenize the context and truncate if it exceeds the max allowed length
    tokens = context.split()
    if len(tokens) > max_length:
        return " ".join(tokens[:max_length])
    return context


def generate_response(query):
    relevant_transcripts = search_weaviate(query)
    context = format_for_openai(relevant_transcripts)
    context = truncate_context(context)  # Ensure context is within token limits
    # return context
    response = ask_openai(query, context)
    return response


def main():
    while True:
        try:

            print("===============================")
            print("QUERY")
            user_query = input()
            if user_query.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break

            response = generate_response(user_query)
            print("")
            print("RESPONSE")
            print(response)
        except KeyboardInterrupt:
            print("\nInterrupted by user, exiting...")
            break
        except Exception as e:
            print("Error:", str(e))
            continue


if __name__ == "__main__":
    main()
