from vector_store import query_vector_store
from llm import generate_response, generate_streaming_response
import os
import sys
import warnings
warnings.filterwarnings("ignore")

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the assistant header."""
    print("\n" + "=" * 50)
    print("🤖 Personal Business Assistant 🤖".center(50))
    print("=" * 50)
    print("Ask me anything about your business ideas and notes.")
    print("Type 'exit' to quit.\n")

def main():
    """Main function to run the CLI assistant."""
    clear_screen()
    print_header()
    
    while True:
        # Get user query
        query = input("🔍 Your question: ")
        
        # Check for exit command
        if query.lower() in ['exit', 'quit', 'bye']:
            print("\nThank you for using your Personal Business Assistant. Goodbye! 👋\n")
            sys.exit(0)
        
        # Skip empty queries
        if not query.strip():
            continue
        
        # Initialize search_results as empty list
        search_results = []
        
        if "read" in query.lower() and "note" in query.lower():
            print("\nSearching your notes... 🔎")
            search_results = query_vector_store(query)
        
        print("\nGenerating response... 🧠")
        
        # Print response header
        print("\n🤖 Answer:")
        print("-" * 50)
        
        # Stream the response token by token
        for token in generate_streaming_response(query, search_results):
            print(token, end="", flush=True)
        
        # Print footer after streaming is complete
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()