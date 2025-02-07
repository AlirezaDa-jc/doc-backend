import requests
import weaviate
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
import json
from transformers import GPT2Tokenizer
import time

app = Flask(__name__)
CORS(app)

client = weaviate.connect_to_local()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


@app.route('/query', methods=['POST'])
def query_weaviate():
    data = request.json.get('input', {})
    user_query = data.get('query')
    limit = int(data.get('limit', 15))

    questions = client.collections.get("files_db")
    try:
        response = questions.query.near_text(
            query=user_query,
            limit=limit
        )
        return jsonify(response.objects)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ollama-stream', methods=['GET'])
def ollama_stream():
    # Get parameters from query string
    user_prompt = request.args.get('prompt')
    limit = request.args.get('limit', 15, type=int)

    # Validate required parameters
    if not user_prompt:
        return Response("Error: 'prompt' parameter is required", status=400)

    # Query Weaviate for context
    questions = client.collections.get("files_db")
    response = questions.query.near_text(
        query=user_prompt,
        limit=limit
    )

    contexts = []
    for obj in response.objects:
        if not hasattr(obj, 'properties'):
            continue

        props = obj.properties
        if 'content' in props:
            contexts.append(props['content'])
        if 'metadata' in props:
            metadata = props['metadata']
            if 'source' in metadata:
                contexts.append(f"Source: {metadata['source']}")
            if 'page' in metadata:
                contexts.append(f"Page: {metadata['page']}")

    context = "".join(contexts)
    query = f"""
    You are an AI assistant focused on providing meaningful responses only to clear, understandable user query. You MUST Follow these rules:
    1. If the user query contains:
       - Random characters (e.g., "asdfgh", "dsadsadsa")
       - Single letters or numbers repeated (e.g., "sssss", "11111")
       - No actual words or questions
       
      STOP and respond ONLY with: "Please provide a clear question or request."
    
    2. If the user query is vague but contains actual words (e.g., "security stuff", "tell me things"):
       Respond with: "Your question is too vague. Could you please be more specific about what you'd like to know?"
    
    3. Only provide detailed responses when the user query contains:
       - Clear questions
       - Specific requests
       - Coherent sentences
    
    4. For appropriate user queries only (if you get past steps 1 2 3):
       - Never mention context, documentation, or sources
       - Never say "based on..." or "it appears..."
       - Give direct answers only
       - Only discuss topics explicitly asked about
    
    Remember: It's better to ask for clarification than to generate an unrelated or unnecessary response.
    DO NOT try to interpret random characters as having meaning.
    DO NOT generate responses about security, technology, or any other topic unless the user query specifically asks about it.
    Only generate a detailed response if the user query contains actual words forming a coherent question or request.
    
    User Query:
    {user_prompt}
    
    Context:
    {context}
    

    
    Response:
    """

    tokens = tokenizer.encode(query)
    if len(tokens) > 1024:
        tokens = tokens[:1024]
        query = tokenizer.decode(tokens)

    def generate():
        payload = {
            "model": "deepseek-v2:latest",
            "prompt": query,
            "stream": True
        }
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        if response_text := data.get("response", ""):
                            yield f"data: {response_text}\n\n"

                        if data.get("done", False):
                            break

                    except json.JSONDecodeError as e:
                        yield f"data: Error parsing response: {str(e)}\n\n"
                        break

        except requests.RequestException as e:
            yield f"data: Error connecting to Ollama: {str(e)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )


@app.route('/prompt', methods=['POST'])
def prompt_weaviate():
    data = request.json.get('input', {})
    user_query = data.get('prompt')
    limit = int(data.get('limit', 15))

    try:
        # Access the collection
        collection = client.collections.get("files_db")

        # Generate response using near_text
        response = collection.generate.near_text(
            query=user_query,  # Vectorize the query automatically
            grouped_task=" Return your response as HTML and dont include tags like html . SCHEMA: html typography tags like <h1-6> <p> <span> etc \n" + user_query,
            limit=limit,
        )

        return jsonify({"response": response.generated})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/test-stream', methods=['GET'])
def test_stream():
    def generate():
        for i in range(10):
            # Format as a Server-Sent Event
            yield f"data: Data chunk {i}\n\n"
            time.sleep(1)  # Simulate delay

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",  # Changed from text/plain
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Transfer-Encoding": "chunked"
        }
    )


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
