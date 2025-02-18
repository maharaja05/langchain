import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

def extract_between(text):

    start_texts = ["AI:","Akasha:"]
    end_texts = ["Human:", "</s>"]
    start_index = -1
    for start_text in start_texts:
        index = text.find(start_text)
        if index != -1:
            start_index = index + len(start_text)
            break

    if start_index == -1:
        return ""  

    end_index = len(text)  
    for end_text in end_texts:
        index = text.find(end_text, start_index)
        if index != -1:
            end_index = index
            break

    next_start_index = len(text)
    for start_text in start_texts:
        index = text.find(start_text, start_index)
        if index != -1:
            next_start_index = min(next_start_index, index)
    
    if next_start_index < end_index:
        end_index = next_start_index

    if start_index < end_index:
        return text[start_index:end_index].strip()
    else:
        return ""

def init():
    load_dotenv()

    st.set_page_config(
        page_title="MultiPersonality Assistant",
        page_icon="🤖"
    )

def question(input,  personality):
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        return_full_text=True,
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
        streaming=True,
    )

    template = ChatPromptTemplate([
            ("system", "You're {personality} personality assistant. Act like {personality} person."),
            ("ai", "Your name is {name}.Greet!"),
            ("human", "{input}"),
        ]
    )

    chain = template | llm

    response = chain.invoke({"personality": personality ,"name": "Akasha","input": input})
    return response

def main():
    init()

    configurations = [
        {"personality": "Thug"},
        {"personality": "Moody"},
        {"personality": "Lazy"},
        {"personality": "Extremly Angry"},
        {"personality": "Extremly Yandere"}
    ]

    tab_labels = [f"{config['personality']} Assistant" for config in configurations]
    tabs = st.tabs(tab_labels)
    
    for i, tab in enumerate(tabs):
        with tab:
            st.header(f"Tab {i + 1}: {configurations[i]['personality']} Assistant")
            input_text = st.text_input(label='Enter prompt' ,key=f"Tab {i + 1}").lower()
            if st.button("Compute", key=f"button_tab_{i}"):
                if input_text:
                    with st.spinner('Thinking....'):
                        response = question(input_text, configurations[i]['personality'])
                        response = extract_between(response)
                    st.text_area(label="Generated Text", value=response, disabled=True)
    

if __name__ == '__main__':
    main()
