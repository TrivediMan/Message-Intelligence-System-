
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"

st.set_page_config(
    page_title="Message Spam Intelligence",
    page_icon="✉️",
    layout="centered"
)

# --------------------------------------------------
# 2. LOAD MODELS SAFELY
# --------------------------------------------------
@st.cache_resource
def load_models():
    loaded_models = {}
    model_files = {
        "KNN": "best_knn.pkl",
        "SVC": "best_svm.pkl",
        "NB": "best_nb.pkl"
    }

    for name, filename in model_files.items():
        model_path = MODEL_DIR / filename

        if not model_path.is_file():
            st.warning(f"{name} model file not found: {model_path}")
            continue

        try:
            loaded_models[name] = joblib.load(model_path)
        except Exception as error:
            st.warning(
                f"{name} could not be loaded. "
                "The saved model may be incompatible with your "
                "installed scikit-learn version. "
                f"Error: {error}"
            )

    return loaded_models


models = load_models()

# --------------------------------------------------
# 3. FEATURES
# --------------------------------------------------
FEATURES = [
    "message_length",
    "word_count",
    "num_urls",
    "num_digits",
    "num_special_chars",
    "spam_keyword_score",
    "legit_keyword_score",
    "sender_activity_score",
    "sender_account_age_days",
    "messages_sent_last_24h",
    "hour_of_day",
    "day_of_week"
]

# --------------------------------------------------
# 4. APP HEADER
# --------------------------------------------------
st.title("✉️ Message Spam Intelligence")
st.write(
    "Enter message and sender features to detect "
    "Spam or Ham using a trained machine learning model."
)

if not models:
    st.error(
        "No models could be loaded. Check the Model folder "
        "and retrain incompatible model files using your "
        "current Python environment."
    )
    st.stop()

# --------------------------------------------------
# 5. MODEL SELECTION
# --------------------------------------------------
st.sidebar.header("Model Settings")

model_name = st.sidebar.selectbox(
    "Choose Classifier",
    list(models.keys())
)

st.sidebar.caption(
    f"Models available: {len(models)}"
)

# --------------------------------------------------
# 6. INPUT FORM
# --------------------------------------------------
st.subheader("Message Features")

with st.form("spam_form"):

    col1, col2 = st.columns(2)

    with col1:
        message_length = st.number_input(
            "Message Length",
            min_value=0,
            value=100
        )

        word_count = st.number_input(
            "Word Count",
            min_value=0,
            value=20
        )

        num_urls = st.number_input(
            "Number of URLs",
            min_value=0,
            value=0
        )

        num_digits = st.number_input(
            "Number of Digits",
            min_value=0,
            value=0
        )

        num_special_chars = st.number_input(
            "Special Characters",
            min_value=0,
            value=2
        )

        spam_keyword_score = st.number_input(
            "Spam Keyword Score",
            min_value=0.0,
            value=0.0
        )

    with col2:
        legit_keyword_score = st.number_input(
            "Legitimate Keyword Score",
            min_value=0.0,
            value=0.0
        )

        sender_activity_score = st.number_input(
            "Sender Activity Score",
            min_value=0.0,
            value=0.5
        )

        sender_account_age_days = st.number_input(
            "Sender Account Age (days)",
            min_value=0,
            value=365
        )

        messages_sent_last_24h = st.number_input(
            "Messages Sent Last 24 Hours",
            min_value=0,
            value=5
        )

        hour_of_day = st.number_input(
            "Hour of Day (0-23)",
            min_value=0,
            max_value=23,
            value=12
        )

        day_of_week = st.number_input(
            "Day of Week (0-6)",
            min_value=0,
            max_value=6,
            value=2
        )

    submitted = st.form_submit_button(
        "Analyze Message",
        use_container_width=True
    )

# --------------------------------------------------
# 7. PREDICTION
# --------------------------------------------------
if submitted:

    input_data = pd.DataFrame(
        [{
            "message_length": message_length,
            "word_count": word_count,
            "num_urls": num_urls,
            "num_digits": num_digits,
            "num_special_chars": num_special_chars,
            "spam_keyword_score": spam_keyword_score,
            "legit_keyword_score": legit_keyword_score,
            "sender_activity_score": sender_activity_score,
            "sender_account_age_days": sender_account_age_days,
            "messages_sent_last_24h": messages_sent_last_24h,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week
        }],
        columns=FEATURES
    )

    model = models[model_name]

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        # Supports common numeric and text class labels.
        normalized_prediction = str(prediction).strip().lower()

        if normalized_prediction in {
            "1", "spam", "true", "yes"
        }:
            st.error("🚨 Spam Message")
        elif normalized_prediction in {
            "0", "ham", "not spam", "false", "no"
        }:
            st.success("✅ Ham (Not Spam)")
        else:
            st.info(f"Predicted class: {prediction}")

        st.caption(f"Selected model: {model_name}")

    except Exception as error:
        st.error(
            "Prediction failed. Check whether the model "
            "was trained using these same 12 input features "
            "in the same order and with compatible preprocessing."
        )
        st.exception(error)
