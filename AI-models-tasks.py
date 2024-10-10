import streamlit as st
from transformers import pipeline
from PIL import Image
from transformers import T5Tokenizer

# Set the page config for the app (only once)
st.set_page_config(page_title="AI models Task", layout="centered")

# Cache model loading to reduce repeated downloading/loading times
@st.cache_resource
def load_translation_model():
    return pipeline("translation_en_to_fr")

@st.cache_resource
def load_object_detection_model():
    return pipeline(model="facebook/detr-resnet-50")

@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_question_answering_model():
    return pipeline(model="deepset/roberta-base-squad2")

@st.cache_resource
def load_text_classification_model():
    return pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_question_generation_model():
    tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    return pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap", tokenizer=tokenizer)

# Define the pages as functions
def page_1():
    st.title("Translation from En to Fr")
    en_fr_translator = load_translation_model()
    user_input = st.text_area("Enter Sentence of English")
    if st.button("Translate"):
        if user_input:
            input = en_fr_translator(user_input)
            translate = f"Translation (EN to FR): {input[0]['translation_text']}"
            st.write(translate)

def page_2():
    st.title("Upload and Display Image for Object Detection")
    detector = load_object_detection_model()
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        output = detector(image)
        output1 = f"Accuracy: {output[0]['score']}, Label: {output[0]['label']}"
        st.write(output1)
    else:
        st.write("Please upload an image.")

def page_3():
    st.title("Text Summarization")
    summarizer = load_summarization_model()
    user_input = st.text_area("Enter any text to summarize")
    if st.button("Summarize"):
        if user_input:
            input = summarizer(user_input, max_length=150, min_length=5, do_sample=False)
            summarize = f"Summary: {input[0]['summary_text']}"
            st.write(summarize)

def page_4():
    st.title("Question Answering")
    oracle = load_question_answering_model()
    user_input = st.text_area("Enter any text for question answering")
    use_input1 = st.text_area("Enter a question to answer")
    if st.button("Answer Question"):
        if user_input and use_input1:
            input = oracle(question=use_input1, context=user_input)
            answer = f"Answer: {input['answer']} (Accuracy: {input['score']})"
            st.write(answer)

def page_5():
    st.title("Text Classification")
    classifier = load_text_classification_model()
    user_input = st.text_area("Enter a sentence to classify as positive or negative")
    if st.button("Classify"):
        if user_input:
            input = classifier(user_input)
            classify1 = f"Classification: {input[0]['label']} (Accuracy: {input[0]['score']})"
            st.write(classify1)

def page_6():
    st.title("Text to Question Generation")
    generator = load_question_generation_model()
    user_input = st.text_area("Enter a sentence to generate a question")
    if st.button("Generate"):
        if user_input:
            prompt = f"generate question: {user_input}"
            try:
                input = generator(prompt)
                generate1 = f"Generated question: {input[0]['generated_text']}"
                st.write(generate1)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.write("Please try again.")

def homepage():
    st.image("https://via.placeholder.com/800x150.png?text=AI+Models+Task", use_column_width=True)
    st.title("Welcome to the AI Task Hub")
    st.write("""
    Explore various AI technologies below. Click on each task to learn more or interact with demos.
    """)

    st.markdown("""
    ### AI Tasks Overview
    - **Translation (EN to FR)**: Converts text from English to French.
    - **Object Detection**: Identifies objects in an image.
    - **Text Summarization**: Reduces the length of a document while retaining key information.
    - **Question Answering**: Responds to questions based on context.
    - **Text Classification**: Categorizes text into predefined classes.
    - **Text Generation**: Produces text based on a given prompt.
    """)

# Dropdown menu in the upper right corner
st.markdown(
    """
    <style>
    .dropdown-container {
        position: absolute;
        top: 10px;
        right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Wrap the dropdown in a div to apply the CSS class
st.markdown('<div class="dropdown-container">', unsafe_allow_html=True)
page = st.selectbox("Select any AI Models", ["Homepage", "Translation from En to Fr", "Object Detection", "Text Summarization",
                                      "Question Answering", "Text Classification", "Questions Generation"])
st.markdown('</div>', unsafe_allow_html=True)

# Run the appropriate page based on selection
if __name__ == "__main__":
    if page == "Homepage":
        homepage()
    elif page == "Translation from En to Fr":
        page_1()
    elif page == "Object Detection":
        page_2()
    elif page == "Text Summarization":
        page_3()
    elif page == "Question Answering":
        page_4()
    elif page == "Text Classification":
        page_5()
    elif page == "Questions Generation":
        page_6()
