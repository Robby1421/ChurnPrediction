import streamlit as st
from pptx import Presentation

# Function to extract slides from a PowerPoint file
def extract_slides(file):
    presentation = Presentation(file)
    slides = []
    for slide in presentation.slides:
        content = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text
                content.append(text)
        slides.append("\n\n".join(content))
    return slides

# Streamlit app layout
st.title("Churn Prediction Use Case")

# File uploader
uploaded_file = st.file_uploader("Upload your PowerPoint File", type="pptx")
if uploaded_file:
    slides_content = extract_slides(uploaded_file)
    total_slides = len(slides_content)
    
    if total_slides > 0:
        # Select the slide to display
        selected_slide = st.slider("Select Slide", 1, total_slides, 1)
        st.write(f"### Slide {selected_slide}:")
        st.write(slides_content[selected_slide - 1])
    else:
        st.warning("No slides were found in the uploaded file.")
else:
    st.info("Upload a PowerPoint file to begin.")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit and python-pptx")
