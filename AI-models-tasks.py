import streamlit as st
from transformers import pipeline
from PIL import Image
from transformers import T5Tokenizer, pipeline
import sentencepiece



# Set the page config for the app (only once)
st.set_page_config(page_title="AI models Task ", layout="centered")

# Define the pages as functions
def page_1():
    st.title("Translation from En to Fr")
    en_fr_translator = pipeline("translation_en_to_fr")
    user_input = st.text_area("Enter Sentence of English")
    if st.button("Translate"):
        if user_input:
            input = en_fr_translator(user_input)
            translate = f"translation_text_en_to_fr: {input[0]['translation_text']}"
            st.write(translate)


def page_2():
    detector = pipeline(model="facebook/detr-resnet-50")
    st.title("Upload and Display Image for Object Detection")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        output = detector(image)
        output1 = f"accuracy: {output[0]['score']}, Label: {output[0]['label']}"
        st.write(output1)
    else:
        st.write("Please upload an image.")


def page_3():
    st.title("Text Summarization")

    # Use a more suitable model for summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    user_input = st.text_area("Enter any text to summarize")
    if st.button("Summarize"):
        if user_input:
            input = summarizer(user_input, max_length=150, min_length=5, do_sample=False)
            summarize = f"Summary: {input[0]['summary_text']}"
            st.write(summarize)


def page_4():
    st.title("Question Answering")
    oracle = pipeline(model="deepset/roberta-base-squad2")
    user_input = st.text_area("Enter any text for question answering")
    use_input1 = st.text_area("Enter any question to answer")
    if st.button("Answer Question"):
        if user_input and use_input1:
            input = oracle(question=use_input1, context=user_input)
            answer = f"answer: {input['answer']} (accuracy: {input['score']})"
            st.write(answer)


def page_5():
    st.title("Text Classification")
    classifier = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    user_input = st.text_area("Enter a sentence to classify as positive or negative")
    if st.button("Classify"):
        if user_input:
            input = classifier(user_input)
            classify1 = f"type of classification: {input[0]['label']} (accuracy: {input[0]['score']})"
            st.write(classify1)




def page_6():
    st.title("Text to Question Generation")
    # Load the tokenizer and the model
    tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    generator = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap", tokenizer=tokenizer)

    user_input = st.text_area("Enter a sentence to generate a question")
    
    if st.button("Generate"):
        if user_input:
            # Add the prefix to indicate question generation
            prompt = f"generate question: {user_input}"
            try:
                # Generate the question based on the input
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

    # Create small points for each AI task
    st.markdown("""
    ### AI Tasks Overview
    - **Translation (EN to FR)**: Converts text from English to French.
    - **Object Detection**: Identifies objects in an image.
    - **Text Summarization**: Reduces the length of a document while retaining key information.
    - **Question Answering**: Responds to questions based on context.
    - **Text Classification**: Categorizes text into predefined classes.
    - **Text Generation**: Produces text based on a given prompt.
    """)

    st.write("Explore each of these tasks in more detail by selecting from the dropdown menu.")


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
