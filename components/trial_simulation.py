import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from io import BytesIO
import base64
from typing import List, Dict, Optional
import json
from dataclasses import dataclass
import time

@dataclass
class TrialArgument:
    content: str
    cited_cases: List[str]
    confidence_score: float
    counter_arguments: List[str]

class TrialSimulator:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.recognizer = sr.Recognizer()
        self.trial_history = []
        
    def initialize_simulation(self, case_type: str, topic: str) -> Dict:
        """Initialize trial simulation with context and parameters."""
        # Generate simulation context using existing documents
        query = f"legal precedents and arguments for {case_type} cases involving {topic}"
        relevant_docs, _, metadata = self.vector_store.search(query, k=5)
        
        context = "\n".join(relevant_docs)
        simulation_params = {
            "case_type": case_type,
            "topic": topic,
            "context": context,
            "stage": "opening_arguments"
        }
        
        return simulation_params
    
    def text_to_speech(self, text: str) -> str:
        """Convert text to speech and return base64 encoded audio."""
        tts = gTTS(text=text, lang='en')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            
        # Read the saved file and encode to base64
        with open(fp.name, 'rb') as fp:
            audio_bytes = fp.read()
        
        # Clean up temporary file
        os.unlink(fp.name)
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        return f'data:audio/mp3;base64,{audio_base64}'
    
    def speech_to_text(self) -> str:
        """Record and convert speech to text."""
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = self.recognizer.listen(source)
            
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            return ""
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return ""
    
    def generate_opposition_argument(self, context: str, user_argument: str) -> TrialArgument:
        """Generate opposition's argument based on context and user's argument."""
        # Prepare prompt for the chat model
        prompt = f"""
        Based on the following context and user's argument, generate a strong counter-argument
        that an opposing counsel might make, including relevant case citations:
        
        Context: {context}
        User's Argument: {user_argument}
        
        Generate a response in JSON format with:
        1. Main argument
        2. List of cited cases
        3. Confidence score (0-1)
        4. Potential counter-arguments to this position
        """
        
        # Get response from OpenAI
        response = get_chat_response(prompt, context)
        
        try:
            response_data = json.loads(response)
            return TrialArgument(
                content=response_data['main_argument'],
                cited_cases=response_data['cited_cases'],
                confidence_score=response_data['confidence_score'],
                counter_arguments=response_data['counter_arguments']
            )
        except json.JSONDecodeError:
            # Fallback parsing if response isn't proper JSON
            return TrialArgument(
                content=response,
                cited_cases=[],
                confidence_score=0.5,
                counter_arguments=[]
            )

def render_trial_simulation():
    st.header("Trial Simulation")
    
    if 'trial_simulator' not in st.session_state:
        if st.session_state.vector_store:
            st.session_state.trial_simulator = TrialSimulator(st.session_state.vector_store)
        else:
            st.warning("Please upload and process legal documents first.")
            return
    
    simulator = st.session_state.trial_simulator
    
    # Initialize or get simulation parameters
    if 'simulation_params' not in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            case_type = st.selectbox(
                "Select Case Type",
                ["Civil", "Criminal", "Administrative", "Constitutional"]
            )
        with col2:
            topic = st.text_input("Enter Case Topic/Subject Matter")
            
        if st.button("Initialize Trial Simulation"):
            st.session_state.simulation_params = simulator.initialize_simulation(
                case_type, topic
            )
            st.rerun()
        return
    
    # Display simulation interface
    st.subheader("Mock Trial Session")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Text", "Speech"],
        horizontal=True
    )
    
    # Get user's argument
    user_argument = ""
    if input_method == "Text":
        user_argument = st.text_area("Present your argument:", height=150)
        submit = st.button("Submit Argument")
    else:
        if st.button("Start Speaking"):
            user_argument = simulator.speech_to_text()
            submit = bool(user_argument)
        else:
            submit = False
    
    # Process argument and generate response
    if submit and user_argument:
        # Generate opposition's response
        opposition_argument = simulator.generate_opposition_argument(
            st.session_state.simulation_params["context"],
            user_argument
        )
        
        # Display response
        st.markdown("### Opposition's Response:")
        
        # Convert to speech if requested
        if st.toggle("Use Text-to-Speech"):
            audio_base64 = simulator.text_to_speech(opposition_argument.content)
            st.markdown(f'<audio autoplay controls><source src="{audio_base64}" type="audio/mp3"></audio>', 
                       unsafe_allow_html=True)
        
        # Display text response
        st.markdown(opposition_argument.content)
        
        # Display cited cases
        if opposition_argument.cited_cases:
            st.markdown("#### Cited Cases:")
            for case in opposition_argument.cited_cases:
                st.markdown(f"- {case}")
        
        # Display potential counter-arguments
        if opposition_argument.counter_arguments:
            with st.expander("View Potential Counter-Arguments"):
                for counter in opposition_argument.counter_arguments:
                    st.markdown(f"- {counter}")
        
        # Add to trial history
        st.session_state.trial_simulator.trial_history.append({
            "user_argument": user_argument,
            "opposition_argument": opposition_argument
        })
    
    # Display trial history
    if st.session_state.trial_simulator.trial_history:
        with st.expander("Trial History"):
            for idx, exchange in enumerate(st.session_state.trial_simulator.trial_history, 1):
                st.markdown(f"### Exchange {idx}")
                st.markdown("**Your Argument:**")
                st.markdown(exchange["user_argument"])
                st.markdown("**Opposition's Response:**")
                st.markdown(exchange["opposition_argument"].content)
                st.markdown("---")
    
    # Reset simulation button
    if st.button("Reset Simulation"):
        del st.session_state.simulation_params
        st.session_state.trial_simulator.trial_history = []
        st.rerun()