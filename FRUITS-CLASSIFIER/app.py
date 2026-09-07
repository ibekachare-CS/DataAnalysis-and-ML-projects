import json
import numpy as np
import streamlit as st
from PIL import Image
import ai_edge_litert.interpreter as tflite

# Setup Page UI
st.set_page_config(page_title="Swahili Fruit Classifier", page_icon="🍌")
st.title("🍌 Swahili Fruit Classifier")
st.write("Pakia picha ya tunda ili kupata jina lake kwa Kiswahili.")

# Load Artifacts
@st.cache_resource
def load_model_and_labels():
    interpreter = tflite.Interpreter(model_path="fruits_cnn_swahili.tflite")
    interpreter.allocate_tensors()

    with open("class_indices.json", "r") as f:
        class_map = json.load(f)

    return interpreter, class_map

try:
    interpreter, class_map = load_model_and_labels()
    st.success("✅ Model imepakiwa kikamilifu!")
except Exception as e:
    st.error(f"Hitilafu ya kupakia model: {e}")
    st.stop()

# File Upload & Processing
uploaded_file = st.file_uploader("Chagua picha...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    st.image(raw_image, caption="Picha Uliyopakia", use_container_width=True)

    resized_img = raw_image.resize((100, 100))
    img_array = np.array(resized_img, dtype=np.float32)

    # IMPORTANT: the retrained model (MobileNetV2 transfer learning) was trained with
    # tf.keras.applications.mobilenet_v2.preprocess_input, which scales pixels to
    # [-1, 1] — NOT the old CNN's [0, 1] scaling (img/255.0). This must match exactly
    # what the training notebook did, or predictions will be silently wrong.
    img_array = (img_array / 127.5) - 1.0

    input_tensor = np.expand_dims(img_array, axis=0)

    if st.button("Tambua Tunda"):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx] * 100)

        # class_indices.json is now correctly built as {"0": "Ndizi", "1": "Chungwa", ...}
        # by the training notebook (enumerate(class_names)), so the string-index lookup
        # is the primary path. The reversed-mapping fallback stays just in case an older
        # or hand-edited class_indices.json ever gets swapped in.
        swahili_label = "Haijulikani"

        if str(top_idx) in class_map:
            swahili_label = class_map[str(top_idx)]
        else:
            idx_to_class_fallback = {int(v): str(k) for k, v in class_map.items() if str(v).isdigit()}
            swahili_label = idx_to_class_fallback.get(top_idx, "Haijulikani")

        st.markdown("---")
        st.subheader("Matokeo:")
        st.metric(label="Aina ya Tunda", value=swahili_label)
        st.metric(label="Uhakika", value=f"{confidence:.2f}%")

        # Debug helper: shows dictionary content if still returning unknown
        with st.expander("Debug Info (Class Map)"):
            st.write("Predicted Index:", top_idx)
            st.write("Loaded JSON Structure:", class_map)