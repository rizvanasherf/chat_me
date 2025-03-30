import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned DialoGPT model
MODEL_NAME = "final_dialogpt_lora"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

st.title("💬 Chat with me")
st.markdown("Type your message below and interact with the chatbot!")

# Initialize session state for conversation history
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None  # Stores encoded chat history
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # Stores conversation for display

# Fallback response for unknown questions
FALLBACK_RESPONSE = "I'm not sure how to answer that. Can you try rephrasing?"

# User input
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        # Encode user input and add EOS token
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

        # Append previous conversation history
        if st.session_state.chat_history_ids is not None:
            bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id
        )

        # Extract only the new response (excluding input text)
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Handle empty responses
        if not bot_response.strip():
            bot_response = FALLBACK_RESPONSE

        # Update conversation history
        st.session_state.chat_history_ids = chat_history_ids
        st.session_state.chat_display.append(("You", user_input))
        st.session_state.chat_display.append(("Bot", bot_response))

# Display chat history
st.subheader("Conversation History")
for sender, message in st.session_state.chat_display:
    st.markdown(f"**{sender}:** {message}")

# Button to clear chat
if st.button("Clear Chat"):
    st.session_state.chat_history_ids = None
    st.session_state.chat_display = []
    st.rerun()