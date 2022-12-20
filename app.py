import sys
import easyocr as ocr
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2


st.set_page_config(
    page_title="easyocr",
    page_icon=" ",
    layout="wide",
    menu_items={
         'Get Help': None,
         'Report a bug': None,
    }
)


@st.cache
def load_model(lang):
    return ocr.Reader([lang], model_storage_directory=".")


def main():
    st.title("Recogniza text from image")
    st.write("For OCR select image from the below supported list")
    image = st.file_uploader(label="Upload Image", type=["png", "jpg", "jpeg"])
    langs = {
        "English": "en",
        "German": "de",
        "Spanish": "es",
    }

    label_langs = st.sidebar.title('Languages')
    feature_choice = st.sidebar.selectbox("Select the language of the text to recognize", list(langs.keys()))
    info_text = st.sidebar.text('Langauages available')
    if image is not None:
        input_image = Image.open(image)
        st.image(input_image)
        reader = load_model(lang=langs.get(feature_choice))
        with st.spinner("Recognition... "):
            result = reader.readtext(np.array(input_image))
            st.title("Result:")
            st.write(pd.DataFrame(result, columns=['bbox','text','conf']))

            # out_str = " "
            # list_text = [ftext[1] for ftext in result]
            # with open(".txt", 'w') as file:
            #     file.write(out_str.join(list_text))
            # result_file = open(".txt", 'r')
            # st.download_button('Recognized text', result_file)
            # st.title("Recognized text in image:")
            # result_text = [text[1] for text in result]
            # st.write(result_text)
    else:
        st.info("Please select a language and upload an image for recognition...")
    st.caption("EASYOCR")



if __name__ == "__main__":
    main()
