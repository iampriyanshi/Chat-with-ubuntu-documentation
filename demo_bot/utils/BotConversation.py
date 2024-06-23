# from utils.Retrieval import GetAnswer
from utils.Utilities import llm_pipeline
from utils.Retrieval import Retrieval as ret

class BotConversation:
    def __init__(self, store, prompt, logger) -> None:
        self.user_input = None
        self.llm = llm_pipeline()
        self.memory = None
        self.retrieval = ret(store, logger)
        self.chat_history = []
        self.prompt = prompt
        self.store={}
        self.logger = logger

    def start_conversation(self):
        print("Starting new conversation ...\n\n")
        self.chat_history = []
        self.continue_conversation()

    def continue_conversation(self):
        # Handle the conversation
        user_input = self.get_user_input("Question: ")
        response = self.retrieval.GetAnswer(user_input, self.chat_history)
        self.chat_history.append(response)

        # Check if the user wants to continue the conversation
        user_input = self.get_user_input("\n\n\nStart new, continue or stop? (start/continue/stop): ")
        if user_input == "start":
            self.start_conversation()
        elif user_input == "continue":
            print("Continuing current conversation ...\n\n")
            self.continue_conversation()
        else:
            print("Stopping conversation ...")
            return
        
    def get_user_input(self, query=None):
        self.user_input = input(query)
        return self.user_input
    
