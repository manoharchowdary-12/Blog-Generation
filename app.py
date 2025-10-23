import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# --- Function to get response from hosted LLaMA model ---
def getLLamaresponse(input_text, no_words, blog_style):
    # Using hosted model instead of local model file
    llm = CTransformers(
        model='TheBloke/Llama-2-7B-Chat-GGML',  # Public HF model
        model_type='llama',
        config={
            'max_new_tokens': 256,
            'temperature': 0.01
        }
    )

    # Prompt Template
    template = """
        Write a blog for {blog_style} job profile on the topic "{input_text}"
        within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )

    # Generate the response from the LLaMA 2 model
    response = llm(prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words
    ))
    
    return response


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Generate Blogs 🤖",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🤖")

# --- User Inputs ---
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
    with st.spinner("Generating blog... please wait ⏳"):
        response = getLLamaresponse(input_text, no_words, blog_style)
        st.subheader("📝 Generated Blog:")
        st.write(response)
