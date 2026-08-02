
responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! What can I do for you?",
    "hey": "Hey! Nice to hear from you.",
    "how are you": "i'm an chatbot, but I'm doing great!",
    "what is your name": "I'm ChatBot, your friendly rule-based assistant.",
    "help": "I can respond to greetings and simple questions. Try saying 'hello' or 'how are you'.",
}

# Commands that will end the conversation
EXIT_COMMANDS = {"bye", "exit", "quit"}


def get_response(user_input: str) -> str:
    """
    Look up the cleaned user input in the knowledge base.
    Returns a default fallback message if no match is found.
    """
    return responses.get(user_input, "I do not understand. Could you rephrase that?")


def run_chatbot() -> None:
    """
    Main chatbot loop:
    - Continuously reads input
    - Sanitizes it
    - Checks for exit commands
    - Otherwise looks up a response and prints it
    """
    print("ChatBot: Hello! I'm your rule-based assistant. Type 'bye' or 'exit' to quit.")

    while True:
        raw_input_text = input("You: ")
        clean_input = raw_input_text.lower().strip()

        if not clean_input:
            # Ignore empty input, ask again
            continue

        if clean_input in EXIT_COMMANDS:
            print("ChatBot: Goodbye! Have a great day!")
            break

        reply = get_response(clean_input)
        print(f"ChatBot: {reply}")


if __name__ == "__main__":
    run_chatbot()