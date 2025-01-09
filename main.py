import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle


# Function to generate caption
def generate_caption(image_path, model, tokenizer, feature_extractor, max_length=34, img_size=224):
    # Preprocess the image
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    image_features = feature_extractor.predict(img, verbose=0)  # Extract image features

    # Generate the caption
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


# Streamlit app interface
def main():
    st.title("Image Caption Generator")
    st.write("Upload one or more images and generate captions using the trained model.")

    # Sidebar for image upload
    st.sidebar.title("Upload Images")
    uploaded_images = st.sidebar.file_uploader(
        "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    # Paths for the saved models and tokenizer
    model_path = "models/model.keras"  # Replace with the actual path
    tokenizer_path = "models/tokenizer.pkl"  # Replace with the actual path
    feature_extractor_path = "models/feature_extractor.keras"  # Replace with the actual path

    # Load the trained models and tokenizer
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    if uploaded_images:
        cols = st.columns(2)  # Two columns for image and caption display

        for idx, uploaded_image in enumerate(uploaded_images):
            # Save each uploaded image temporarily
            image_path = f"uploaded_image_{idx}.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            # Generate caption for the image
            caption = generate_caption(image_path, caption_model, tokenizer, feature_extractor)

            # Display the image and its caption with styled text
            with cols[idx % 2]:  # Alternate between the two columns
                st.image(image_path, use_container_width=True)
                st.markdown(
                    f"""<p style="font-size:18px;"><b style="color:black;">Caption:</b> <span style="color:red;">{caption}</span></p>""",
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
