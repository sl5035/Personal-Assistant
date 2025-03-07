from vector_store import query_vector_store
from llm import generate_response
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
        
        print("\nSearching your notes... 🔎")
        
        # try:
        # Query vector store for relevant documents
        search_results = query_vector_store(query)
        
        print("\nGenerating response... 🧠")
        
        # Generate response using LLM
        response = generate_response(query, search_results)
        
        # Print response
        print("\n🤖 Answer:")
        print("-" * 50)
        print(response)
        print("-" * 50 + "\n")
            
        # except Exception as e:
            # print(f"\n❌ Document Query Error: {str(e)}")
        
        # Prompt to continue
        input("\nPress Enter to ask another question...")
        clear_screen()
        print_header()

if __name__ == "__main__":
    main()