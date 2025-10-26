import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers


# --- Function to get response from LLaMA model ---
def getLLamaresponse(input_text, no_words, blog_style):
    # Use hosted Hugging Face model (not local .bin)
    llm = CTransformers(
        model='TheBloke/Llama-2-7B-Chat-GGML',  # hosted model
        model_type='llama',
        config={
            'max_new_tokens': 256,
            'temperature': 0.01
        }
    )

    # Prompt Template
    template = """
        Write a well-structured blog about "{input_text}" for {blog_style}.
        Approximate length: {no_words} words.
        Make sure the content ends with a complete conclusion.

    """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )

    # Create final prompt text
    final_prompt = prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words
    )

    # Generate response (fixed for new version)
    response = llm.invoke(final_prompt)
    return response


# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Generate Blogs 🤖",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🤖")

# --- Inputs ---
input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Researchers', 'Data Scientist', 'Common People'),
        index=0
    )

# --- Generate Button ---
submit = st.button("Generate")

# --- Output Section ---
if submit:
    if not input_text or not no_words:
        st.warning("Please fill all fields before generating!")
    else:
        with st.spinner("Generating blog... please wait ⏳"):
            response = getLLamaresponse(input_text, no_words, blog_style)
            st.subheader("📝 Generated Blog:")
            st.write(response)
