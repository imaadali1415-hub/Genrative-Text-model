import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit App UI
st.title("🧠 Generative Text Model (GPT-2)")
st.subheader("Type a prompt and let the AI complete it!")

# User prompt input
prompt = st.text_input("Enter your prompt:", "Once upon a time")

# Generation settings
max_len = st.slider("Max length of output", 50, 200, 100)
temperature = st.slider("Creativity (temperature)", 0.1, 1.0, 0.7)

if st.button("Generate Text"):
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=max_len,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.markdown("### ✍️ Generated Text:")
    st.write(generated_text)
