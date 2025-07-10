import streamlit as st


from YH.app import main as chatbot1_main, clear_current_conversation as reset_chatbot1
from LN.test import main as chatbot2_main
from HL.app2 import main as chatbot3_main, reset_chat as reset_chatbot3



def setup_chatbot_sessions():
    """ Initialize or reset chatbot sessions """
    if 'active_chatbot' not in st.session_state:
        st.session_state['active_chatbot'] = None

def reset_other_chatbots(current_chatbot):
    """ Reset states of other chatbots when switching """
    if current_chatbot != 'MistralAI':
        reset_chatbot1()
    if current_chatbot != 'GEMINI':
        reset_chatbot3()

def display_chatbot():
    """ Display the active chatbot based on user selection """
    chatbot_map = {
        'MistralAI': chatbot1_main,
        'Meta LLaMA': chatbot2_main,
        'GEMINI': chatbot3_main
    }

    current_chatbot = st.session_state['active_chatbot']

    if current_chatbot in chatbot_map:
        reset_other_chatbots(current_chatbot)
        chatbot_map[current_chatbot]()
    else:
        st.markdown("Select a chatbot from the sidebar to begin interacting.")

def main():
    setup_chatbot_sessions()

    st.sidebar.title("Chatbot Selector")
    chatbot_options = ['Select a Chatbot', 'MistralAI', 'Meta LLaMA', 'GEMINI']
    selected_chatbot = st.sidebar.selectbox("Choose a chatbot", chatbot_options)

    # Update active chatbot only when a valid chatbot is selecteds
    if selected_chatbot != 'Select a Chatbot':
        st.session_state['active_chatbot'] = selected_chatbot

    
    display_chatbot()

if __name__ == "__main__":
    main()
