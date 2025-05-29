import streamlit as st
# import replicate
import os
from requestchatbot import ChatbotClient
from query_data import query_finetuned_rag, query_finetuned, query_rag, query_base

from dotenv import load_dotenv




load_dotenv()
# App title
st.set_page_config(page_title="💬 Friends Chatbot")

selected_option = 'Gemma 2B'
st.session_state.gemmachat= ChatbotClient()
def clear_chat_history():
    if selected_option == 'Gemma 2B':
        st.session_state.gemmachat.clean_chat()
    st.session_state.messages = [{"role": "assistant", "content": "Hôm nay tôi có thể hỗ trợ gì cho bạn?"}]

with st.sidebar:
    st.title('💬 Sale Chatbot')
    st.write('Chatbot này cung cấp một số biến thể của mô hình Llama 33 LLM + RAG. Hãy thoải mái để tôi giúp bạn tư vấn mua máy tính')
    
    
    replicate_api_token_env = os.environ.get("REPLICATE_API_TOKEN")
    
    # Obtain Credentials
    # if 'REPLICATE_API_TOKEN' in st.secrets:
    #     st.success('API key already provided!', icon='✅')
    #     replicate_api = st.secrets['REPLICATE_API_TOKEN']
    
    
    if replicate_api_token_env:
        
        st.success('API key already provided!', icon='✅')
        replicate_api =replicate_api_token_env
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    os.environ['OPENAI_API_KEY'] = replicate_api_token_env # this is for the embedding model
    
    selected_option = st.sidebar.selectbox(
        'Choose a model:', 
        ['Gemma 2B','LLaMA3 with RAG'],
        key = 'model',
        on_change=clear_chat_history
    )

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)



# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hôm nay tôi có thể hỗ trợ gì cho bạn?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLaMA2 response
def find_product(prompt_input):
    pass
def generate_response(prompt_input):
    string_dialogue = "Bạn là một trợ lý hữu ích. Bạn hãy trả lời câu hỏi như một trợ lý bán hàng. \n"
    
    
    # below is for session messages history; we disabled this feature here for test run
    # for dict_message in st.session_state.messages:
    #    if dict_message["role"] == "user":
    #        string_dialogue += "User: " + dict_message["content"] + "\n\n"
    #    else:
    #        string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    
    # run selected model
    # 'LLaMA2', 'Finetuned LLaMA2', 'LLaMA2 with RAG', 'Finetuned LLaMA2 with RAG'
    output = ""
    if selected_option == 'LLaMA3 with RAG':
        output = query_rag(f"{string_dialogue} {prompt_input} Trợ lý: ")
    else:
        output = st.session_state.gemmachat.chat(prompt_input)
    
    return output


# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


