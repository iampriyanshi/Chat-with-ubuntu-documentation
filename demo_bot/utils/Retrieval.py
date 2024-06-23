## TODO Add imports
from langchain_community.document_transformers import (
    LongContextReorder,
)
from utils.Utilities import llm_pipeline
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

## TODO Add below:
# 1. errors/exception handling 
# 2. loggers
def set_up_retriever(store):
    base_retriever=store.as_retriever()
    base_retriever.search_kwargs={'k': 20}
    return base_retriever

class Retrieval:
    def __init__(self, store, logger) -> None:
        self.llm = llm_pipeline()
        self.retriever = set_up_retriever(store=store)
        self.logger = logger

    def ReorderDocuments(self, documents):
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(documents)
        return reordered_docs

    def get_history_aware_retriever(self):
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", '{input}'),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        self.logger.info("History-aware retriever initialized.")
        return history_aware_retriever

    def get_question_answer_chain(self):

        ### Answer question ###
        system_prompt = (
            """
            Using the relevant section of the following pieces of context, provide only the necessary steps or answers without any explanations or additional commentary. 
            Avoid repetition and ensure introductory statements are direct and simple. 
            Limit your response to the most essential information only, avoid repetition, and ensure introductory and closing statements are direct and simple.
            Don't restate the context. If you don't know the answer, say that you don't know.
            \n\n
            Context: \n\n
            {context}
            """
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.logger.info("Question-Answer chain initialized.")
        return question_answer_chain

    def get_conversation_chain(self):
        history_aware_retriever = self.get_history_aware_retriever()
        question_answer_chain = self.get_question_answer_chain()
        rag_chain = create_retrieval_chain(history_aware_retriever | self.ReorderDocuments, question_answer_chain).pick("answer")

        self.logger.info("Conversation chain initialized.")
        return rag_chain

    def GetAnswer(self, user_input, chat_history):
        conversation = self.get_conversation_chain()
        response = ''
        for chunk in conversation.stream({"input": user_input, "chat_history": chat_history}):
            response += chunk
            print(f"{chunk}", end="")
        
        return response
