import os
import itertools
from google import genai # Ensure this import is correct based on your setup

# Load API keys from environment variables
API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
API_KEY_2 = os.getenv("GEMINI_API_KEY_2")

initialized_clients = []

if API_KEY_1:
    try:
        client1 = genai.Client(api_key=API_KEY_1)
        initialized_clients.append(client1)
        print("INFO: Gemini API Client 1 initialized successfully.")
    except Exception as e:
        print(f"WARNING: Failed to initialize Gemini API Client 1. Error: {e}")
        print("         Please ensure GEMINI_API_KEY_1 is valid and has access.")

if API_KEY_2:
    try:
        client2 = genai.Client(api_key=API_KEY_2)
        initialized_clients.append(client2)
        print("INFO: Gemini API Client 2 initialized successfully.")
    except Exception as e:
        print(f"WARNING: Failed to initialize Gemini API Client 2. Error: {e}")
        print("         Please ensure GEMINI_API_KEY_2 is valid and has access.")

if not initialized_clients:
    raise RuntimeError(
        "FATAL: No Gemini API clients could be initialized. "
        "Please set valid GEMINI_API_KEY_1 and/or GEMINI_API_KEY_2 environment variables."
    )
elif len(initialized_clients) < 2:
    print(
        f"WARNING: Only {len(initialized_clients)} API client was initialized. "
        "Round-robin will use the available client(s), but to fully leverage two keys, ensure both are set up correctly."
    )


# Create a cycle iterator for round-robin if there are clients
client_cycler = itertools.cycle(initialized_clients) if initialized_clients else None

def get_next_api_client():
    """
    Returns the next available genai.Client instance in a round-robin fashion.
    """
    if not client_cycler:
        raise RuntimeError("FATAL: No API clients available for cycling.")
    
    selected_client = next(client_cycler)
    
    # Optional: For debugging, identify which client is being used.
    # This requires a bit more logic if clients are identical beyond their key.
    # client_id = "Client 1" if selected_client == initialized_clients[0] else "Client 2" if len(initialized_clients) > 1 and selected_client == initialized_clients[1] else "Unknown Client"
    # print(f"DEBUG: Using API {client_id}")
    return selected_client
