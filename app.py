import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Smart City Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart City Traffic Prediction")
st.write(
    "An AI-based traffic congestion prediction system "
    "using machine learning and data analytics."
)

st.info(
    "Demonstration mode: This application uses simulated "
    "traffic data and does not connect to live traffic sensors."
)


# -------------------------------
# GENERATE SAMPLE TRAFFIC DATA
# -------------------------------

@st.cache_data
def generate_traffic_data():
    np.random.seed(42)

    n = 2000

    hour = np.random.randint(0, 24, n)
    day_of_week = np.random.randint(0, 7, n)

    # More vehicles during morning and evening peak hours
    peak_hour = (
        ((hour >= 7) & (hour <= 10)) |
        ((hour >= 17) & (hour <= 20))
    )

    vehicle_count = np.random.randint(50, 500, n)

    vehicle_count = (
        vehicle_count + peak_hour.astype(int) * 120
    )

    vehicle_count = np.clip(
        vehicle_count, 50, 650
    )

    average_speed = (
        75 - vehicle_count * 0.085
        + np.random.normal(0, 8, n)
    )

    average_speed = np.clip(
        average_speed, 5, 90
    )

    is_rainy = np.random.randint(0, 2, n)

    # Create a congestion score for demonstration.
    # This is a simulated rule, not a real-world measurement.
    congestion_score = (
        vehicle_count * 0.7
        - average_speed * 1.2
        + peak_hour.astype(int) * 50
        + is_rainy * 15
        + np.random.normal(0, 20, n)
    )

    congestion = np.where(
        congestion_score < 200,
        "LOW",
        np.where(
            congestion_score < 330,
            "MODERATE",
            "HIGH"
        )
    )

    return pd.DataFrame({
        "hour": hour,
        "day_of_week": day_of_week,
        "vehicle_count": vehicle_count,
        "average_speed": average_speed.round(2),
        "is_rainy": is_rainy,
        "congestion": congestion
    })


df = generate_traffic_data()


# -------------------------------
# TRAIN MACHINE LEARNING MODEL
# -------------------------------

features = [
    "hour",
    "day_of_week",
    "vehicle_count",
    "average_speed",
    "is_rainy"
]

X = df[features]
y = df["congestion"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# -------------------------------
# DASHBOARD METRICS
# -------------------------------

st.subheader("📊 Traffic Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Sample Records",
    len(df)
)

col2.metric(
    "Model Test Accuracy",
    f"{accuracy * 100:.2f}%"
)

col3.metric(
    "Traffic Categories",
    df["congestion"].nunique()
)


# -------------------------------
# TRAFFIC DATA ANALYSIS
# -------------------------------

st.subheader("📈 Traffic Data Analysis")

left, right = st.columns(2)

with left:
    st.write("Average Vehicle Count by Hour")

    hourly_traffic = (
        df.groupby("hour")["vehicle_count"]
        .mean()
    )

    st.bar_chart(hourly_traffic)

with right:
    st.write("Distribution of Congestion Levels")

    congestion_counts = (
        df["congestion"]
        .value_counts()
        .reindex(
            ["LOW", "MODERATE", "HIGH"],
            fill_value=0
        )
    )

    st.bar_chart(congestion_counts)


# -------------------------------
# USER INPUT FOR PREDICTION
# -------------------------------

st.subheader("🔮 Predict Traffic Congestion")

st.write(
    "Enter traffic conditions to predict the congestion category."
)

with st.form("traffic_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        selected_hour = st.slider(
            "Hour of the Day",
            min_value=0,
            max_value=23,
            value=9
        )

        selected_day = st.selectbox(
            "Day of the Week",
            options=list(range(7)),
            format_func=lambda x: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][x]
        )

        vehicle_count_input = st.slider(
            "Number of Vehicles",
            min_value=50,
            max_value=650,
            value=300
        )

    with col2:
        average_speed_input = st.slider(
            "Average Vehicle Speed (km/h)",
            min_value=5,
            max_value=90,
            value=35
        )

        rainy_input = st.selectbox(
            "Weather Condition",
            options=[0, 1],
            format_func=lambda x:
                "Rainy" if x == 1 else "Clear"
        )

    submitted = st.form_submit_button(
        "Predict Congestion"
    )


if submitted:

    input_data = pd.DataFrame([{
        "hour": selected_hour,
        "day_of_week": selected_day,
        "vehicle_count": vehicle_count_input,
        "average_speed": average_speed_input,
        "is_rainy": rainy_input
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == "LOW":
        st.success("🟢 Predicted Congestion: LOW")

    elif prediction == "MODERATE":
        st.warning("🟡 Predicted Congestion: MODERATE")

    else:
        st.error("🔴 Predicted Congestion: HIGH")

    st.write(
        f"Model confidence: {confidence:.2f}%"
    )

    st.caption(
        "Confidence is the model's estimated probability "
        "for its selected category, not a guarantee of accuracy."
    )

    probability_df = pd.DataFrame({
        "Congestion Level": model.classes_,
        "Estimated Probability": probabilities
    })

    st.write("Estimated Probability by Category")

    st.bar_chart(
        probability_df.set_index("Congestion Level")
    )


# -------------------------------
# MODEL EVALUATION
# -------------------------------

with st.expander("View Model Evaluation"):

    st.write("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df.round(3))

    st.write("Confusion Matrix")

    labels = model.classes_

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    st.dataframe(cm_df)

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    st.pyplot(fig)
    plt.close(fig)


# -------------------------------
# VIEW SAMPLE DATA
# -------------------------------

with st.expander("View Sample Traffic Dataset"):

    st.dataframe(df.head(100))

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Traffic Dataset as CSV",
        data=csv_data,
        file_name="traffic_data.csv",
        mime="text/csv"
    )


# -------------------------------
# FOOTER
# -------------------------------

st.markdown("---")

st.caption(
    "Academic prototype | AI + IoT concept + Data Analytics | "
    "Traffic data is simulated for demonstration purposes."
)