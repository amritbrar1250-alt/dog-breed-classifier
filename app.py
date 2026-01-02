import json
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ----------------------------
# Page config + styling
# ----------------------------
st.set_page_config(page_title="Dog Breed Classifier", layout="wide")

CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.card {
  border-radius: 16px;
  padding: 1rem 1.2rem;
  border: 1px solid rgba(120,120,120,0.25);
  background: rgba(250,250,250,0.55);
}
.small { font-size: 0.92rem; color: rgba(40,40,40,0.82); }
.muted { color: rgba(40,40,40,0.70); }
.hr { height: 1px; background: rgba(120,120,120,0.25); margin: 1rem 0; }
.kpi {
  display: inline-block;
  padding: 0.4rem 0.65rem;
  border-radius: 999px;
  border: 1px solid rgba(120,120,120,0.25);
  background: rgba(255,255,255,0.7);
  margin-right: 0.4rem;
  font-size: 0.9rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

IMG_SIZE = (224, 224)

# ----------------------------
# Load model + labels
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    model = tf.keras.models.load_model("models/dogs_best.keras", compile=False)
    with open("models/dog_labels.json", "r") as f:
        labels = json.load(f)
    # Warm-up for consistent latency
    dummy = tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
    _ = model(dummy, training=False)
    return model, labels

model, label_names = load_model_and_labels()

# ----------------------------
# Helpers
# ----------------------------
def pretty_label(label: str) -> str:
    return label.split("-", 1)[-1].replace("_", " ").title()

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def list_example_images(example_dir="examples"):
    if not os.path.isdir(example_dir):
        return []
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = [f for f in os.listdir(example_dir) if f.lower().endswith(exts)]
    files.sort()
    return [os.path.join(example_dir, f) for f in files]

def predict_topk(pil_img: Image.Image, top_k: int):
    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]
    idx = probs.argsort()[-top_k:][::-1]
    names = [pretty_label(label_names[i]) for i in idx]
    vals = [float(probs[i]) for i in idx]
    return idx, names, vals, probs

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### Settings")
    show_chart = st.toggle("Show probability chart", True)
    top_k = st.slider("Top-K predictions", 3, 10, 5, 1)
    low_conf_threshold = st.slider(
    "Low-confidence threshold (%)",
    min_value=5,
    max_value=60,
    value=25,
    step=5)
    st.markdown("---")
    st.markdown("### Model details")
    st.markdown('<div class="small">Backbone: EfficientNetB0 (transfer learning)</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Dataset: Stanford Dogs (120 breeds)</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Test accuracy: 84.56%</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Inference runs locally on the server</div>', unsafe_allow_html=True)
    
# ----------------------------
# Header
# ----------------------------
st.markdown("## Dog Breed Classifier")
st.markdown('<div class="muted">Upload a photo or pick an example to get top predictions with confidence scores.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="kpi">120 breeds</div>'
    '<div class="kpi">EfficientNet transfer learning</div>'
    '<div class="kpi">Top-K predictions</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ----------------------------
# Input: Upload OR Examples (fixed)
# ----------------------------
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Upload"   # or "Examples"
if "example_path" not in st.session_state:
    st.session_state.example_path = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

left, right = st.columns([0.46, 0.54], gap="large")

selected_img = None
selected_source = None

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Choose an input")

    mode = st.radio(
        "Input source",
        ["Upload", "Examples"],
        horizontal=True,
        index=0 if st.session_state.input_mode == "Upload" else 1
    )
    st.session_state.input_mode = mode

    if mode == "Upload":
        uploaded = st.file_uploader(
            "Choose an image (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            key="uploader_key"
        )
        if uploaded is not None:
            st.session_state.uploaded_file = uploaded
<<<<<<< HEAD
            selected_img = ImageOps.exif_transpose(Image.open(path))
=======
            selected_img = ImageOps.exif_transpose(Image.open(uploaded))
>>>>>>> 6c579db (Fix image orientation using EXIF metadata)
            selected_source = "Uploaded image"

        st.markdown('<div class="small">Tip: Use a clear photo with the dog centered and well lit.</div>', unsafe_allow_html=True)

    else:
        examples = list_example_images("examples")
        if not examples:
            st.markdown('<div class="small">No example images found. Create an "examples/" folder and add JPG/PNG files.</div>', unsafe_allow_html=True)
        else:
            labels = [os.path.basename(p) for p in examples]
            choice = st.selectbox("Pick an example image", labels, index=0)
            path = examples[labels.index(choice)]

            use_btn = st.button("Use this example")
            if use_btn:
                st.session_state.example_path = path

            # Only use the example if the user clicked the button at least once
            if st.session_state.example_path is not None:
                selected_img = Image.open(st.session_state.example_path)
                selected_source = f"Example: {os.path.basename(st.session_state.example_path)}"

    st.markdown("</div>", unsafe_allow_html=True)

    if selected_img is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Preview")
        st.markdown(f'<div class="small">{selected_source}</div>', unsafe_allow_html=True)
        st.image(selected_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
# ----------------------------
# Results
# ----------------------------
with right:
    st.markdown("### Choose an input")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if selected_img is None:
        st.info("Upload an image or select an example to see predictions.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        with st.spinner("Running inference..."):
            idx, names, vals, full_probs = predict_topk(selected_img, top_k)

        best_name = names[0]
        best_prob = vals[0]
        threshold = low_conf_threshold / 100.0

        if best_prob < threshold:
            st.warning(
                f"Low confidence result ({best_prob*100:.2f}%). "
                "Try a clearer photo or zoom in on the dog."
            )
            st.markdown(
                "- Make sure the dog is centered and in focus\n"
                "- Use good lighting (avoid heavy shadows)\n"
                "- Reduce background clutter\n"
                "- Try a photo where the face/body is visible"
            )

        st.markdown(f"**Best prediction:** {best_name}")
        st.markdown(f'<div class="small">Confidence: {best_prob*100:.2f}%</div>', unsafe_allow_html=True)
        st.markdown("")

        st.markdown("#### Top predictions")
        for name, p in zip(names, vals):
            st.write(f"{name} — {p*100:.2f}%")
            st.progress(min(max(p, 0.0), 1.0))

        if show_chart:
            st.markdown("#### Probability chart")
            st.bar_chart({n: v for n, v in zip(names, vals)})

        # Downloadable JSON results (nice “product” touch)
        result_payload = {
            "source": selected_source,
            "top_k": top_k,
            "predictions": [
                {"rank": r + 1, "breed": n, "probability": float(v)}
                for r, (n, v) in enumerate(zip(names, vals))
            ],
        }
        st.download_button(
            label="Download results (JSON)",
            data=json.dumps(result_payload, indent=2),
            file_name="dog_breed_predictions.json",
            mime="application/json",
        )

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown('<div class="small">Built with TensorFlow, EfficientNet transfer learning, and Streamlit.</div>', unsafe_allow_html=True)
