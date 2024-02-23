import streamlit as st
from clarifai.client.model import Model
import base64
from PIL import Image

# openai_api_key = os.getenv("OPEN_AI")

# Function to generate image using Clarifai DALL-E model
def generate_image(user_description):
    prompt = f"You are a professional comic artist. Based on the below user's description and content, create a proper story comic: {user_description}"
    inference_params = dict(quality="standard", size="1024x1024")
    model_prediction = Model(
        f"https://clarifai.com/openai/dall-e/models/dall-e-3"
    ).predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    output_base64 = model_prediction.outputs[0].data.image.base64
    with open("generated_image.png", "wb") as f:
        f.write(output_base64)
    return "generated_image.png"

# Function to understand image and generate story using Clarifai GPT-4 Vision model
def understand_image(base64_image):
    prompt = "Analyze the content of this image and write a creative, engaging story that brings the scene to life. Describe the characters, setting, and actions in a way that would captivate a young audience:"
    inference_params = dict(temperature=0.2, image_base64=base64_image)
    model_prediction = Model(
        "https://clarifai.com/openai/chat-completion/models/gpt-4-vision"
    ).predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    return model_prediction.outputs[0].data.text.raw

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to convert text to speech
def text_to_speech(input_text):
    inference_params = dict(voice="alloy", speed=1.0)
    model_prediction = Model(
        "https://clarifai.com/openai/tts/models/openai-tts-1"
    ).predict_by_bytes(
        input_text.encode(), input_type="text", inference_params=inference_params
    )
    audio_base64 = model_prediction.outputs[0].data.audio.base64
    return audio_base64

def main():
    st.set_page_config(page_title="Interactive Media Creator", layout="wide")
    st.title("StoryVerse AI")
    st.write("""
    * Generate captivating stories accompanied by stunning visuals and immersive audio
    """)

    # Initialize session variable 'terms' to keep track of free usage
    if 'terms' not in st.session_state:
        st.session_state['terms'] = 0
        st.session_state['is_usr_key'] = False

    # Check if user has exceeded free usage
    if st.session_state['terms'] >= 3:
        st.sidebar.header("Enter Your CLARIFAI PAT Key")
        clarifai_pat_usr = st.sidebar.text_input("CLARIFAI PAT Key")
        if clarifai_pat_usr:
            clarifai_pat = clarifai_pat_usr
            st.session_state['is_usr_key'] = True  # Store user-entered key
        else:
            st.sidebar.warning("Enter your PAT key to continue!", icon="⚠️")
    else:
        clarifai_pat = st.secrets["CLARIFAI_PAT"]

    with st.sidebar:
        st.header("Controls")
        image_description = st.text_area("Description for Image Generation", height=100)
        
        # Check if user has exceeded free usage
        if st.session_state['terms'] < 4 or st.session_state['is_usr_key']:
            generate_image_btn = st.button("Generate Story!")
            if generate_image_btn:
                st.session_state['terms'] += 1
                # print(st.session_state["terms"])

    col1, col2 = st.columns(2)

    with col1:
        st.header("Comic Art")
        
        # Check if user has exceeded free usage
        if (st.session_state['terms'] < 4 or st.session_state['is_usr_key']) and generate_image_btn and image_description:
            with st.spinner("Generating image..."):
                image_path = generate_image(image_description)
                if image_path:
                    st.image(
                        image_path,
                        caption="Generated Comic Image",
                        use_column_width=True,
                    )
                    st.success("Image generated!")
                else:
                    st.error("Failed to generate image.")
        elif (st.session_state['terms'] >= 3 and not st.session_state['is_usr_key']):
            st.error("You have reached the maximum try limit!\nEnter your PAT key to continue.")

    with col2:
        st.header("Story")
        
        # Check if user has exceeded free usage
        if (st.session_state['terms'] < 4 or st.session_state['is_usr_key']) and generate_image_btn and image_description:
            with st.spinner("Creating a story..."):
                base64_image = encode_image(image_path)
                understood_text = understand_image(base64_image)
                audio_base64 = text_to_speech(understood_text)
                st.audio(audio_base64, format="audio/mp3")
                st.success("Audio generated from image understanding!")
    
    if st.session_state["terms"] != 0:
        st.write(
                """
                Made By **_Jaweria Batool_**
                """
            )

            # link to GitHub README file
        st.write("For more information about how the app works, please check out the [GitHub README](https://github.com/Jaweria-B/Media-App) file.")

        

if __name__ == "__main__":
    main()