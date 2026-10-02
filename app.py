import gradio as gr
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# Load the trained ANN model
model = load_model("breast_cancer_ann.keras")

# Load the scaler used during training
scaler = joblib.load("scaler.pkl")


# Prediction function
def predict(
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    concave_points_mean,
    symmetry_mean,
    fractal_dimension_mean,
    radius_se,
    texture_se,
    perimeter_se,
    area_se,
    smoothness_se,
    compactness_se,
    concavity_se,
    concave_points_se,
    symmetry_se,
    fractal_dimension_se,
    radius_worst,
    texture_worst,
    perimeter_worst,
    area_worst,
    smoothness_worst,
    compactness_worst,
    concavity_worst,
    concave_points_worst,
    symmetry_worst,
    fractal_dimension_worst
):

    # Store the 30 input values
    values = [[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,
        radius_se,
        texture_se,
        perimeter_se,
        area_se,
        smoothness_se,
        compactness_se,
        concavity_se,
        concave_points_se,
        symmetry_se,
        fractal_dimension_se,
        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst
    ]]

    # Feature names must be in the same order as training
    columns = [
        'radius_mean',
        'texture_mean',
        'perimeter_mean',
        'area_mean',
        'smoothness_mean',
        'compactness_mean',
        'concavity_mean',
        'concave points_mean',
        'symmetry_mean',
        'fractal_dimension_mean',
        'radius_se',
        'texture_se',
        'perimeter_se',
        'area_se',
        'smoothness_se',
        'compactness_se',
        'concavity_se',
        'concave points_se',
        'symmetry_se',
        'fractal_dimension_se',
        'radius_worst',
        'texture_worst',
        'perimeter_worst',
        'area_worst',
        'smoothness_worst',
        'compactness_worst',
        'concavity_worst',
        'concave points_worst',
        'symmetry_worst',
        'fractal_dimension_worst'
    ]

    # Convert inputs into a DataFrame
    input_data = pd.DataFrame(values, columns=columns)

    # Apply the same scaling used during training
    input_scaled = scaler.transform(input_data)

    # Get prediction probability
    probability = model.predict(input_scaled, verbose=0)[0][0]

    # Convert probability into a class
    if probability >= 0.5:
        diagnosis = "Malignant"
    else:
        diagnosis = "Benign"

    return diagnosis, f"{probability:.2%}"


# Create the Gradio interface
demo = gr.Interface(
    fn=predict,

    inputs=[
        gr.Number(label="Radius Mean"),
        gr.Number(label="Texture Mean"),
        gr.Number(label="Perimeter Mean"),
        gr.Number(label="Area Mean"),
        gr.Number(label="Smoothness Mean"),
        gr.Number(label="Compactness Mean"),
        gr.Number(label="Concavity Mean"),
        gr.Number(label="Concave Points Mean"),
        gr.Number(label="Symmetry Mean"),
        gr.Number(label="Fractal Dimension Mean"),

        gr.Number(label="Radius SE"),
        gr.Number(label="Texture SE"),
        gr.Number(label="Perimeter SE"),
        gr.Number(label="Area SE"),
        gr.Number(label="Smoothness SE"),
        gr.Number(label="Compactness SE"),
        gr.Number(label="Concavity SE"),
        gr.Number(label="Concave Points SE"),
        gr.Number(label="Symmetry SE"),
        gr.Number(label="Fractal Dimension SE"),

        gr.Number(label="Radius Worst"),
        gr.Number(label="Texture Worst"),
        gr.Number(label="Perimeter Worst"),
        gr.Number(label="Area Worst"),
        gr.Number(label="Smoothness Worst"),
        gr.Number(label="Compactness Worst"),
        gr.Number(label="Concavity Worst"),
        gr.Number(label="Concave Points Worst"),
        gr.Number(label="Symmetry Worst"),
        gr.Number(label="Fractal Dimension Worst")
    ],

    outputs=[
        gr.Textbox(label="Diagnosis"),
        gr.Textbox(label="Malignancy Probability")
    ],

    title="🩺 Breast Cancer Classification using ANN",

    description=(
        "Enter the 30 tumor measurements to classify "
        "the tumor as Benign or Malignant using a trained "
        "Artificial Neural Network."
    )
)


# Launch the application
demo.launch()