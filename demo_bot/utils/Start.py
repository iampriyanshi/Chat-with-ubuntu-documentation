from utils.BotConversation import BotConversation as bot_conv

def start_app(store, templates, logger):
    print("Started Demo Bot.\n")
    bot = bot_conv(store, templates, logger)
    bot.start_conversation()
            






    
