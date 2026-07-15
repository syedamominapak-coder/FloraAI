import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="FloraAI",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else None,
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
# Professional mobile-card layout: green background, a card with a
# solid green header (title + subtitle), an image panel, a prediction
# line, and two full-width action buttons stacked vertically.

st.markdown("""
<style>

#MainMenu, header, footer {visibility: hidden;}

.stApp{
    background: linear-gradient(180deg, #ffeef2 0%, #ffd6e0 100%);
    min-height:100vh;
}

.block-container{
    max-width: 980px;
    margin: 0 auto;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

/* card shell */
.app-card{
    background:#ffffff;
    border-radius: 20px;
    overflow:hidden;
    box-shadow: 0 16px 36px rgba(0,0,0,0.22);
    margin-bottom: 18px;
}

/* standalone navbar, sits above the content card like a website nav */
.navbar{
    background:#d6336c;
    border-radius: 14px;
    padding: 20px 32px;
    display:flex;
    align-items:center;
    gap:16px;
    box-shadow: 0 10px 24px rgba(214,51,108,0.28);
    margin-bottom: 24px;
}

.navbar .logo-badge{
    background:#ffffff;
    border-radius: 50%;
    padding: 5px;
    width:52px;
    height:52px;
    flex-shrink:0;
    display:flex;
    align-items:center;
    justify-content:center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

.navbar .logo-badge img{
    width:100%;
    height:100%;
    object-fit:contain;
    border-radius:50%;
}

.navbar .header-text{
    text-align:left;
}

.navbar .title{
    color:#ffffff;
    font-size:22px;
    font-weight:700;
    letter-spacing:0.2px;
    margin:0;
}

.navbar .subtitle{
    color:rgba(255,255,255,0.85);
    font-size:13px;
    font-weight:400;
    margin-top:2px;
}

/* two-column body: image/upload on the left, results on the right */
.app-body{
    padding: 28px 32px 30px 32px;
}

.image-frame{
    background:#f3f4f6;
    border-radius: 14px;
    overflow:hidden;
    border: 1px solid #e5e7eb;
    margin-bottom: 14px;
}

.placeholder-box{
    background:#f3f4f6;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    height: 320px;
    display:flex;
    align-items:center;
    justify-content:center;
    margin-bottom:14px;
}

.placeholder-box svg{
    width:72px;
    height:72px;
    opacity:0.35;
}

.results-panel{
    display:flex;
    flex-direction:column;
    justify-content:center;
}

.prediction-text{
    text-align:left;
    color:#6b7280;
    font-size:15px;
    margin-bottom: 4px;
}

.prediction-text b{
    color:#1f2937;
    font-weight:700;
    font-size:22px;
}

.confidence-text{
    text-align:left;
    color:#9ca3af;
    font-size:13px;
    margin-bottom: 22px;
}

/* button row inside results panel */
div[data-testid="stButton"]{
    width:100%;
}

div[data-testid="stButton"] > button{
    width:100%;
    display:flex;
    align-items:center;
    justify-content:center;
    background: linear-gradient(135deg, #f9a8d4 0%, #ec4899 100%);
    color:#ffffff;
    border:none;
    border-radius: 12px;
    padding: 16px 24px;
    min-height: 52px;
    font-weight:600;
    font-size:16px;
    margin-bottom:12px;
    transition: background 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
    box-shadow: 0 6px 14px rgba(236,72,153,0.32);
}

div[data-testid="stButton"] > button:hover{
    background: linear-gradient(135deg, #ec4899 0%, #d6336c 100%);
    color:#ffffff;
    box-shadow: 0 8px 18px rgba(214,51,108,0.42);
    transform: translateY(-1px);
}

div[data-testid="stButton"] > button:active{
    background: linear-gradient(135deg, #d6336c 0%, #b8215c 100%);
    color:#ffffff;
    transform: translateY(0px);
}

div[data-testid="stButton"] > button:focus:not(:active){
    color:#ffffff;
}

[data-testid="stFileUploader"] section,
[data-testid="stCameraInput"] video{
    border-radius: 10px;
}

.footer-text{
    text-align:center;
    color:#9d174d;
    margin-top:10px;
    font-size:12px;
    opacity:0.75;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_flower_model():
    return tf.keras.models.load_model("models/flower_cnn_model.keras")

model = load_flower_model()

class_names = [
    "Daisy",
    "Dandelion",
    "Roses",
    "Sunflowers",
    "Tulips"
]

# ---------------- SESSION STATE ----------------

if "mode" not in st.session_state:
    st.session_state.mode = None  # None, "camera", "upload"

def set_mode(mode):
    st.session_state.mode = mode

# ---------------- STANDALONE NAVBAR ----------------

def get_base64_image(path):
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO_PATH = "assets/logo.png"

if os.path.exists(LOGO_PATH):
    logo_b64 = get_base64_image(LOGO_PATH)
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="logo">'
else:
    logo_html = ""

st.markdown(
    f"""
    <div class="navbar">
        <div class="logo-badge">{logo_html}</div>
        <div class="header-text">
            <p class="title">Detect Flowers</p>
            <p class="subtitle">Custom TensorFlow CNN</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- CARD: BODY ----------------

st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.markdown('<div class="app-body">', unsafe_allow_html=True)

left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    # ---------------- IMAGE INPUT AREA ----------------

    image = None

    if st.session_state.mode == "camera":
        camera_image = st.camera_input("Take a photo", label_visibility="collapsed")
        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")

    elif st.session_state.mode == "upload":
        uploaded_file = st.file_uploader(
            "Camera Roll",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    # ---------------- DISPLAY IMAGE OR PLACEHOLDER ----------------

    if image is not None:
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="placeholder-box">
                <svg viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="1.5"
                     xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14M14 8h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                          stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            """,
            unsafe_allow_html=True
        )

with right_col:
    st.markdown('<div class="results-panel">', unsafe_allow_html=True)

    # ---------------- PREDICTION ----------------

    probabilities = None

    if image is not None:

        img = image.resize((180, 180))
        img_array = np.expand_dims(np.array(img), 0)

        prediction = model.predict(img_array, verbose=0)
        probabilities = tf.nn.softmax(prediction[0]).numpy()

        index = np.argmax(probabilities)
        flower = class_names[index]
        confidence = probabilities[index] * 100

        st.markdown(
            f'<div class="prediction-text">Prediction<br><b>{flower}</b></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="confidence-text">{confidence:.1f}% confidence</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="prediction-text">No image selected yet</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="confidence-text">Use the buttons below to get started</div>',
            unsafe_allow_html=True
        )

    # ---------------- ACTION BUTTONS ----------------

    if st.button("Take a photo"):
        set_mode("camera")
        st.rerun()

    if st.button("Camera Roll"):
        set_mode("upload")
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # close results-panel

st.markdown('</div>', unsafe_allow_html=True)  # close app-body

st.markdown('</div>', unsafe_allow_html=True)  # close app-card

# ---------------- OPTIONAL: FULL PROBABILITY BREAKDOWN ----------------

if probabilities is not None:
    with st.expander("View confidence for all classes"):
        for name, prob in zip(class_names, probabilities):
            percent = float(prob * 100)
            st.write(f"**{name}**: {percent:.2f}%")
            st.progress(percent / 100)

# ---------------- FOOTER ----------------

st.markdown(
    '<div class="footer-text">Built with TensorFlow, Keras and Streamlit</div>',
    unsafe_allow_html=True
)
