import os
from openai import OpenAI
from typing import List, Dict
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageParam

def validate_openai_api_key() -> tuple[bool, str]:
    """Validate OpenAI API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "OpenAI API key not found in environment variables"
    try:
        client = OpenAI(api_key=api_key)
        # Make a minimal API call to validate the key
        client.models.list()
        return True, "OpenAI API key is valid"
    except Exception as e:
        if "authentication" in str(e).lower():
            return False, "Invalid OpenAI API key"
        elif "rate limit" in str(e).lower():
            return False, "OpenAI API rate limit exceeded"
        return False, f"Error validating OpenAI API key: {str(e)}"

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI's API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [list(embedding.embedding) for embedding in response.data]
    except Exception as e:
        if "authentication" in str(e).lower():
            raise Exception("Invalid OpenAI API key")
        elif "rate limit" in str(e).lower():
            raise Exception("OpenAI API rate limit exceeded. Please try again later")
        else:
            raise Exception(f"Error generating embeddings: {str(e)}")

def get_chat_response(prompt: str, context: str) -> str:
    """Get chat completion from OpenAI with context."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    try:
        client = OpenAI(api_key=api_key)
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": "You are a legal assistant. Use the provided context to answer questions accurately and professionally."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {prompt}"
            }
        ]
        
        response: ChatCompletion = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        if response.choices and response.choices[0].message.content:
            return str(response.choices[0].message.content)
        return "No response generated"
    except Exception as e:
        if "authentication" in str(e).lower():
            raise Exception("Invalid OpenAI API key")
        elif "rate limit" in str(e).lower():
            raise Exception("OpenAI API rate limit exceeded. Please try again later")
        else:
            raise Exception(f"Error getting chat response: {str(e)}")
