from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from gtts import gTTS  # new import
from io import BytesIO  # new import

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from utils import *
import os
os.environ['CURL_CA_BUNDLE'] = ''

st.subheader("RevAI")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

import os
os.environ["OPENAI_API_KEY"] = "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"


llm = ChatOpenAI(model_name="gpt-3.5-turbo")

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)



i=0
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

import pyttsx3
with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)

            def text_to_speech(text):
                audio_bytes = BytesIO()
                tts = gTTS(text=text, lang="en")
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                return audio_bytes.read()
                    

            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.audio(text_to_speech(response), format="audio/wav")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
        

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          