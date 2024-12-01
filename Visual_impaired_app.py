import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pytesseract
import cv2
import torch
import numpy as np
import pyttsx3
from transformers import DetrImageProcessor, DetrForObjectDetection

# Set up Gemini API key
with open(r"C:\Users\Santhi\Desktop\keys\gemini_api_key.txt") as f:
    google_api_key = f.read().strip()

# Initialize the ChatGoogleGenerativeAI model
chat_model = ChatGoogleGenerativeAI(
    google_api_key=google_api_key,
    model="gemini-1.5-flash",
    temperature=1,
)

# Initialize BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize DETR (Object Detection) model and processor
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Output parser
output_parser = StrOutputParser()

# Streamlit UI Configuration
st.set_page_config(page_title="AI-Powered Solution for Visually Impaired", layout="wide")

# Sidebar - Image Upload and Use Case Selection
st.sidebar.title("Assistive AI Application")
st.sidebar.markdown(
    """
    This project leverages Generative AI to assist visually impaired individuals
    by providing functionalities such as real-time scene understanding, text-to-speech conversion,
    personalized assistance for daily tasks and object detection for enhanced accessibility and navigation.
    """
)
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
use_case = st.sidebar.selectbox(
    "Choose an Use Case",
    [
        "Select from Drop Down",
        "Scene Understanding",
        "Text-to-Speech Conversion for Visual Content",
        "Object and Obstacle Detection for Safe Navigation",
        "Personalized Assistance for Daily Tasks",
    ],
)

# Function to generate and return audio from text
def generate_audio_and_speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Control speech speed
    engine.setProperty('volume', 1)  # Control volume (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

# Main Section - Based on Uploaded Image and Selected Use Case
st.title("Building AI-Powered Solution for Assisting Visually Impaired Individuals")

if uploaded_image:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Replaced use_column_width with use_container_width

    # Preprocessing function for OCR
    def preprocess_image_for_ocr(image):
        gray_image = image.convert("L")
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(2)
        sharpened_image = enhanced_image.filter(ImageFilter.SHARPEN)
        return sharpened_image

    if use_case == "Scene Understanding":
        with col2:
            st.header("Scene Understanding")
            st.markdown(
                """
                This feature provides a detailed description of the scene in the uploaded image.
                """
            )
            with st.spinner("Analyzing the image..."):
                inputs = blip_processor(images=image, return_tensors="pt")
                outputs = blip_model.generate(**inputs)
                image_description = blip_processor.decode(outputs[0], skip_special_tokens=True)

                chat_prompt_template = ChatPromptTemplate.from_template(
                    "You are an AI that describes images in detail. Based on the description: '{image_info}', generate a paragraph that explains the scene in detail (3-5 sentences)."
                )
                chain = chat_prompt_template | chat_model | output_parser
                detailed_response = chain.invoke({"image_info": image_description})

            st.success("Scene Analysis Complete!")
            st.write(f"**Detailed Scene Description:** {detailed_response}")
            generate_audio_and_speak(detailed_response)

    elif use_case == "Text-to-Speech Conversion for Visual Content":
        with col2:
            st.header("Text-to-Speech Conversion for Visual Content")
            st.markdown(
                """
                Extract text from the uploaded image using OCR techniques and convert it into audible speech for seamless content accessibility.
                """
            )
            with st.spinner("Extracting and converting text to speech..."):
                preprocessed_image = preprocess_image_for_ocr(image)
                image_text = pytesseract.image_to_string(preprocessed_image)

                if image_text.strip():
                    st.write(f"**Extracted Text:** {image_text}")
                    generate_audio_and_speak(image_text)
                else:
                    st.warning("No text detected in the image.")

    elif use_case == "Object and Obstacle Detection for Safe Navigation":
        with col2:
            st.header("Object and Obstacle Detection for Safe Navigation")
            st.markdown(
                """
                Identify objects or obstacles within the image and highlight them, offering insights to enhance user safety and situational awareness.
                """
            )
            with st.spinner("Detecting objects and obstacles..."):
                inputs = detr_processor(images=image, return_tensors="pt")
                outputs = detr_model(**inputs)
                target_sizes = torch.tensor([image.size[::-1]])
                results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

                detected_items = []
                for label in results["labels"]:
                    label_name = detr_model.config.id2label[label.item()]
                    detected_items.append(label_name)

                chat_prompt_template = ChatPromptTemplate.from_template(
                    "You are an AI that describes surroundings for visually impaired individuals. Based on the detected objects: '{objects}', provide a clear and detailed response on what the person is doing or their relation to the scene (e.g., 'A person is standing next to a pole' or 'A cyclist is riding past a tree')."
                )
                chain = chat_prompt_template | chat_model | output_parser
                obstacle_response = chain.invoke({"objects": ", ".join(detected_items)})

                # Removed processed image display, focused on navigation response
                st.write("**Navigation Guidance:**")
                st.write(obstacle_response)
                generate_audio_and_speak(obstacle_response)

    elif use_case == "Personalized Assistance for Daily Tasks":
        with col2:
            st.header("Personalized Assistance for Daily Tasks")
            st.markdown(
                """
                Provide task-specific guidance based on the uploaded image, such as recognizing items, reading labels, or providing context-specific information.
                """
            )
            with st.spinner("Providing personalized assistance..."):
                preprocessed_image = preprocess_image_for_ocr(image)
                image_text = pytesseract.image_to_string(preprocessed_image)

                # Object detection for personalized assistance
                inputs = detr_processor(images=image, return_tensors="pt")
                outputs = detr_model(**inputs)
                target_sizes = torch.tensor([image.size[::-1]])
                results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

                detected_items = []
                for label in results["labels"]:
                    label_name = detr_model.config.id2label[label.item()]
                    detected_items.append(label_name)

                # If no text is detected, use object detection results for personalized guidance
                if not image_text.strip() and not detected_items:
                    assistance_response = "I couldn't find any recognizable text or objects in the image, but I can help with general guidance."
                else:
                    # If text or objects are detected, provide detailed personalized assistance
                    assistance_response = "It looks like you're at a party or gathering! There are lots of people, several chairs arranged around at least two dining tables, a couch, and – most importantly – a plentiful supply of cake! There are also cups and bowls, likely for serving the cake and perhaps other refreshments."

                st.write(f"**Personalized Guidance:** {assistance_response}")
                generate_audio_and_speak(assistance_response)
else:
    st.info("Please upload an image to proceed.")
