import streamlit as st
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="NeurOm Chatbot", layout="wide")
st.title("NeurOm Chatbot") # <<< MODIFIED TITLE

# --- Initialize session state variables ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_initial_options" not in st.session_state:
    st.session_state.show_initial_options = True
if "llm_instance" not in st.session_state:
    try:
        st.session_state.llm_instance = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", temperature=0.1,
        )
        print("LLM Initialized Successfully (Session State).")
    except Exception as e:
        print(f"LLM Init Error: {e}")
        st.session_state.llm_instance = None
if "retriever_instance" not in st.session_state:
    st.session_state.retriever_instance = None
# --- End of session state initialization ---

def get_current_time_info():
    now = datetime.now()
    current_time_str = now.strftime("%I:%M %p")
    current_hour = now.hour
    return current_time_str, current_hour

@st.cache_resource
def load_and_prepare_retriever():
    print("Attempting to load and prepare retriever (cached function)...")
    pdf_files = [
        "CerboTech Chatbot doc (3).pdf",
        "The_Miracle_of_Mindfulness__An_Introductio_-_Thich_Nhat_Hanh.pdf",
        "zenmind.pdf",
        "Mindfulness_in_Plain_English.pdf",
        "Kathleen_McDonald_Robina_Courtin_How_to.pdf",
        "Daniel Goleman_ Richard J. Davidson - The Science of Meditation_ How to Change Your Brain, Mind and Body .pdf"
    ]
    all_pages_from_all_pdfs = []
    loaded_at_least_one_pdf = False
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found at '{pdf_path}'. Skipping this file.")
            continue
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            all_pages_from_all_pdfs.extend(pages)
            print(f"Successfully loaded {len(pages)} pages from {pdf_path}")
            loaded_at_least_one_pdf = True
        except Exception as e:
            print(f"Error loading PDF '{pdf_path}': {e}")

    if not loaded_at_least_one_pdf:
        print("No PDF documents were successfully loaded for the knowledge base.")
        return None
    if not all_pages_from_all_pdfs:
        print("No content extracted from PDFs after loading.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs_chunks = text_splitter.split_documents(all_pages_from_all_pdfs)
    if not docs_chunks:
        print("PDF content might be empty or could not be split into meaningful chunks.")
        return None
    print(f"Total documents after splitting: {len(docs_chunks)}")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(documents=docs_chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 7})
        print("Retriever initialized successfully (cached function).")
        return retriever
    except Exception as e:
        print(f"Error creating vector store or retriever: {e}.")
        return None

def generate_llm_response(user_query, retriever_obj, llm_obj, is_initial_mood_selection=False):
    if retriever_obj is None: return "I'm sorry, my knowledge base isn't available right now. Please check if the PDF files are correctly loaded."
    if llm_obj is None: return "I'm sorry, the language model isn't available right now. Please check the API key and configuration."

    user_query_lower = user_query.lower()
    severe_distress_keywords = [
        "i want to die", "suicidal", "thinking about hurting myself", "kill myself",
        "end my life", "no hope", "hopeless", "can't cope anymore", "crippling anxiety"
    ]
    is_severe_distress = any(keyword in user_query_lower for keyword in severe_distress_keywords)
    if "depressed" in user_query_lower and \
       ("can't function" in user_query_lower or "can't do anything" in user_query_lower or \
        "overwhelming sadness all the time" in user_query_lower or "want to give up" in user_query_lower):
        is_severe_distress = True

    if is_severe_distress:
        return (
            "I hear that you're going through a very difficult time, and I want you to know that your feelings are valid. "
            "While NeurOm offers tools for general well-being, for what you're describing, "
            "it's really important to talk to someone who can offer professional support. "
            "If you're feeling severely depressed or overwhelmed, or if you are in crisis, "
            "I strongly encourage you to reach out to a psychiatrist, therapist, or another qualified mental health professional immediately. "
            "Please remember, you don't have to go through this alone. There are people who can support you."
        )

    current_time_string, current_hour_int = get_current_time_info()
    time_specific_guidance = ""
    prompt_instruction_for_mood = ""
    general_meditation_alternative = "exploring the 'BreatheEasy' techniques in the NeurOm app, listening to some calming 'Music' from our selection, or perhaps trying a general mindfulness practice from the provided texts"
    general_music_alternative = "our general 'Music' selection in the NeurOm app for calm or focus (available any time), or perhaps a quiet moment for reflection as suggested in the mindfulness books"
    is_time_specific_feature_query = False

    if "morning meditation" in user_query_lower:
        is_time_specific_feature_query = True
        start_hour, end_hour = 4, 12
        if start_hour <= current_hour_int < end_hour:
            time_specific_guidance = f"It's currently {current_time_string}. This is a great time for NeurOm's Morning Meditation (best between 4 AM and 12 PM)!"
        else:
            time_specific_guidance = (f"Regarding NeurOm's Morning Meditation, it's currently {current_time_string}. "
                                      f"This activity is best performed between 4 AM and 12 PM. "
                                      f"Since it's outside this window, perhaps you'd like to try {general_meditation_alternative} instead?")
    elif "night music" in user_query_lower or ("sleep" in user_query_lower and "music" in user_query_lower and ("neurom" in user_query_lower or "app" in user_query_lower)):
        is_time_specific_feature_query = True
        start_hour, end_hour_next_day = 20, 3
        is_time = (current_hour_int >= start_hour) or (current_hour_int < end_hour_next_day)
        if is_time:
            time_specific_guidance = (f"It's currently {current_time_string}. "
                                      f"This is a perfect time for NeurOm's Night Music (best between 8 PM and 3 AM) to help you unwind for sleep.")
        else:
            time_specific_guidance = (f"Regarding NeurOm's Night Music, it's currently {current_time_string}. "
                                      f"This audio is best listened to between 8 PM and 3 AM. "
                                      f"At this time, you might enjoy {general_music_alternative} instead?")

    if not is_time_specific_feature_query:
        mood_keywords = ["stressed", "anxious", "overwhelmed", "bored", "unfocused", "distracted", "down", "sad", "tired", "low energy", "depressed", "relax", "calm"]
        goal_keywords = ["improve memory", "logical thinking", "improve reflexes", "lung capacity", "panic"]
        if is_initial_mood_selection or \
           any(mood in user_query_lower for mood in mood_keywords) or \
           any(goal in user_query_lower for goal in goal_keywords):
            prompt_instruction_for_mood = ("The user expressed/selected a mood or goal: '{user_input_for_mood}'. "
                                           "Primarily consult the 'MOOD/GOAL TO ACTIVITY/GAME MAPPING' section from the 'CerboTech Chatbot doc' in the context "
                                           "to suggest one or two suitable NeurOm app games or activities. "
                                           "If the query is very general about mindfulness or meditation not specific to NeurOm, you can draw from other provided books. "
                                           "Briefly state why each suggestion might be helpful.")

    system_prompt_template_string = (
        "You are an assistant for the NeurOm mental well-being app and a guide to mindfulness practices based on provided texts. "
        "The user's current time is {current_time}. "
        "{time_guidance} {mood_instruction} "
        "Your primary role is to guide users to suitable games or activities within the NeurOm app OR suggest relevant mindfulness practices from the provided books, based on their query and the retrieved context. "
        "When suggesting for a mood/goal, prioritize NeurOm app features from the 'CerboTech Chatbot doc' using its 'MOOD/GOAL TO ACTIVITY/GAME MAPPING'. "
        "Only refer to general practices from other books if the query is explicitly about them or very general and not covered by NeurOm features. "
        "If you don't know the answer from the context, or if the context doesn't fully address the user's query, "
        "say that you don't know or provide the best information you can. "
        "Keep answers concise, aiming for a maximum of three to four sentences."
        "\n\n"
        "Context:\n{context}"
    )
    user_input_for_mood_placeholder = user_query if prompt_instruction_for_mood else ""
    final_system_prompt_message = system_prompt_template_string.format(
        current_time=current_time_string,
        time_guidance=time_specific_guidance,
        mood_instruction=prompt_instruction_for_mood.format(user_input_for_mood=user_input_for_mood_placeholder) if prompt_instruction_for_mood else "",
        context="{context}"
    )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [("system", final_system_prompt_message), ("human", "{input}")]
    )
    try:
        question_answer_chain = create_stuff_documents_chain(llm_obj, chat_prompt_template)
        rag_chain = create_retrieval_chain(retriever_obj, question_answer_chain)
        response = rag_chain.invoke({"input": user_query})
        return response["answer"]
    except Exception as e:
        print(f"Error in generate_llm_response: {e}")
        return "I'm sorry, I encountered an issue trying to answer that. Please try again."

# --- Main app flow ---
if st.session_state.retriever_instance is None:
    with st.spinner("Initializing NeurOm's knowledge... Please wait a moment."):
        st.session_state.retriever_instance = load_and_prepare_retriever()
    if st.session_state.retriever_instance is None:
        st.error("Failed to load the knowledge base. The chatbot may not function correctly. Please check PDF files and console for errors.")

llm_instance = st.session_state.llm_instance
retriever_instance = st.session_state.retriever_instance

if llm_instance is None: # Check if LLM failed to load and show persistent error if so
    st.error("Failed to initialize the Language Model. Please check your GOOGLE_API_KEY and internet connection. The chatbot will not be able to respond.")
    # You might want to st.stop() here if the app is completely unusable without LLM

# Add initial greeting message if chat history is empty
if not st.session_state.messages:
    initial_greeting = "Hi there! I'm your NeurOm guide. It's great to connect with you!"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

# Display all chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Display initial option buttons IF show_initial_options is True ---
if st.session_state.show_initial_options:
    st.markdown("---")
    st.markdown("**How can I help you get started today?**")

    options = {
        "Help me Relax and De-stress": "I want to relax and de-stress",
        "Challenge My Mind": "I want a mental challenge",
        "Boost My Focus": "I want to boost my focus",
        "Lift My Spirits or Find Calm": "I want to lift my spirits or find some calm",
        "I'll Type My Own Question": "something_else"
    }

    # <<< --- CORRECTED: Buttons will stack vertically by default --- >>>
    for option_text, option_query in options.items():
        button_key = f"btn_{option_text.lower().replace(' ', '_').replace('/', '_')}" # More robust key
        if st.button(option_text, key=button_key, use_container_width=True):
            st.session_state.show_initial_options = False # Hide buttons after selection

            if option_query != "something_else":
                st.session_state.messages.append({"role": "user", "content": option_query})
                if llm_instance and retriever_instance:
                    with st.spinner("NeurOm is thinking..."):
                        assistant_response = generate_llm_response(option_query, retriever_instance, llm_instance, is_initial_mood_selection=True)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                elif not retriever_instance:
                    st.session_state.messages.append({"role": "assistant", "content": "I'm having trouble accessing my knowledge base right now."})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "I'm having trouble thinking right now. Please check the setup."})
            st.rerun()
    st.markdown("---")
# <<< --- END OF VERTICAL BUTTON CORRECTION --- >>>

# --- Get user input using st.chat_input ---
if user_query_from_input := st.chat_input("Or, type your question or how you're feeling here..."):
    st.session_state.show_initial_options = False
    st.session_state.messages.append({"role": "user", "content": user_query_from_input})
    st.rerun() # Rerun to display user message immediately

# --- Process the latest user message to generate bot response ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]

    if llm_instance and retriever_instance:
        with st.chat_message("assistant"):
            with st.spinner("NeurOm is thinking..."):
                is_initial = False
                initial_option_queries = [
                    "I want to relax and de-stress", "I want a mental challenge",
                    "I want to boost my focus", "I want to lift my spirits or find some calm"
                ]
                if user_query in initial_option_queries and len(st.session_state.messages) <= 2 : # Check if it's likely an initial button press
                    is_initial = True
                assistant_response = generate_llm_response(user_query, retriever_instance, llm_instance, is_initial_mood_selection=is_initial)
            st.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    elif not retriever_instance and llm_instance: # Check if only retriever failed
        if st.session_state.messages[-1]["role"] == "user":
            error_msg = "I'm sorry, my knowledge base isn't available. Please check the PDF file setup."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun() # Rerun to display this error message
    elif not llm_instance and retriever_instance: # Check if only LLM failed
        if st.session_state.messages[-1]["role"] == "user":
            error_msg = "I'm sorry, the language model isn't available. Please check the API key setup."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()
    elif not llm_instance and not retriever_instance: # Both missing
        if st.session_state.messages[-1]["role"] == "user":
            error_msg = "I'm sorry, I'm having trouble starting up. Please check the setup."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()