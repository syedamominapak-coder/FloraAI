import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Flower AI Classifier",
    page_icon="🌸",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#172554,#1e293b);
}

.block-container{
    max-width:900px;
    padding-top:1.2rem;
}

.title{
    text-align:center;
    font-size:55px;
    font-weight:800;
    background:linear-gradient(90deg,#ff4b91,#ffb347,#7c4dff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:#d1d5db;
    font-size:19px;
    margin-bottom:25px;
}

.prediction-box{
    background:#1f2937;
    padding:22px;
    border-radius:18px;
    border:1px solid #374151;
}

.footer{
    text-align:center;
    color:#9ca3af;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🌸 Flower AI")

st.sidebar.markdown("---")

st.sidebar.success("Deep Learning Project")

st.sidebar.metric("Validation Accuracy", "77.9%")

st.sidebar.metric("Flower Classes", "5")

st.sidebar.markdown("### Dataset")
st.sidebar.info("TensorFlow Flowers Dataset")

st.sidebar.markdown("### Model")
st.sidebar.write("Custom CNN")

st.sidebar.markdown("---")

st.sidebar.write("### Flower Classes")

st.sidebar.write("🌼 Daisy")
st.sidebar.write("🌿 Dandelion")
st.sidebar.write("🌹 Roses")
st.sidebar.write("🌻 Sunflowers")
st.sidebar.write("🌷 Tulips")

st.sidebar.markdown("---")
st.sidebar.caption("TensorFlow • Keras • Streamlit")

# ---------------- LOAD MODEL ----------------

model = tf.keras.models.load_model(
    "models/flower_cnn_model.keras"
)

class_names = [
    "Daisy",
    "Dandelion",
    "Roses",
    "Sunflowers",
    "Tulips"
]

# ---------------- HEADER ----------------

st.markdown(
'<div class="title">🌸 Flower AI Classifier</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Upload a flower image and let the CNN recognize it instantly.</div>',
unsafe_allow_html=True
)

# ---------------- BANNER ----------------

if os.path.exists("assets/flowerbanner.jpg"):
    st.image(
        "assets/flowerbanner.jpg",
        use_container_width=True
    )

st.write("")

# ---------------- FILE UPLOADER ----------------

uploaded_file = st.file_uploader(
    "📤 Upload Flower Image",
    type=["jpg","jpeg","png"]
)

# ---------------- PREDICTION ----------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((180,180))

    img = np.array(img)

    img = np.expand_dims(img,0)

    prediction = model.predict(img,verbose=0)

    probabilities = tf.nn.softmax(
        prediction[0]
    ).numpy()

    index = np.argmax(probabilities)

    flower = class_names[index]

    confidence = probabilities[index]*100

    st.markdown("---")

    st.markdown(
        '<div class="prediction-box">',
        unsafe_allow_html=True
    )

    col1,col2 = st.columns(2)

    with col1:

        st.metric(
            "🌼 Prediction",
            flower
        )

    with col2:

        st.metric(
            "🎯 Confidence",
            f"{confidence:.2f}%"
        )

    st.success(
        f"The CNN predicts this flower is **{flower}**."
    )

    st.markdown("</div>",unsafe_allow_html=True)

    st.write("")

    st.subheader("📊 Confidence for All Flower Classes")

    for name,prob in zip(class_names,probabilities):

        percent = float(prob*100)

        st.write(f"**{name}**")

        st.progress(percent/100)

        st.write(f"{percent:.2f}%")

    st.write("")

    st.subheader("📈 Prediction Probability Chart")

    fig,ax = plt.subplots(figsize=(7,4))

    ax.bar(
        class_names,
        probabilities*100
    )

    ax.set_ylabel("Confidence (%)")

    ax.set_ylim(0,100)

    plt.xticks(rotation=20)

    st.pyplot(fig)

st.markdown(
"""
<div class="footer">
Made with ❤️ using TensorFlow, Keras & Streamlit
</div>
""",
unsafe_allow_html=True
)