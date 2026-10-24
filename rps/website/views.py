import gc
import traceback

import numpy as np
from PIL import Image, UnidentifiedImageError

def get_tf():
    import tensorflow as tf
    return tf
    
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST


# ============================================================
# HOME PAGE
# ============================================================

def camera_page(request):

    return render(request, "camera.html" )


# ============================================================
# MODELS ARE LOADED ONE AT A TIME INSIDE predict()
# ============================================================


# ============================================================
# GET MODEL INPUT SIZE
# ============================================================

def get_model_input_size(model):

    input_shape = model.input_shape


    # --------------------------------------------------------
    # Some Keras models can return a list of input shapes.
    # --------------------------------------------------------

    if isinstance(
        input_shape,
        list
    ):

        input_shape = input_shape[0]


    # --------------------------------------------------------
    # Expected:
    #
    # (None, height, width, channels)
    #
    # --------------------------------------------------------

    if (
        input_shape is None
        or len(input_shape) != 4
    ):

        raise ValueError(
            f"Unsupported model input shape: "
            f"{input_shape}"
        )


    height = input_shape[1]

    width = input_shape[2]

    channels = input_shape[3]


    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if height is None:

        height = 224


    if width is None:

        width = 224


    if channels != 3:

        raise ValueError(
            f"Model expects {channels} channels. "
            f"Expected RGB 3-channel image."
        )


    return (
        int(height),
        int(width)
    )


# ============================================================
# PREPARE IMAGE FOR MODEL
# ============================================================

def prepare_image_for_model(
    image,
    model
):

    # --------------------------------------------------------
    # Get model's required size
    # --------------------------------------------------------

    height, width = (
        get_model_input_size(model)
    )


    print(
        f"Preparing image: "
        f"{width}x{height}"
    )


    # --------------------------------------------------------
    # Resize according to THIS model
    # --------------------------------------------------------

    resized_image = image.resize(
        (width, height)
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Preprocessing
    #
    # If your trained model already contains
    # ResNet50 preprocess_input, keep:
    #
    # MODEL_INCLUDES_PREPROCESSING = True
    #
    # Otherwise preprocessing is applied here.
    # --------------------------------------------------------

    if not settings.MODEL_INCLUDES_PREPROCESSING:

        image_array = (
            get_tf().keras.applications
            .resnet50
            .preprocess_input(
                image_array
            )
        )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ============================================================
# CONVERT MODEL OUTPUT TO PROBABILITIES
# ============================================================

def convert_to_probabilities(
    prediction
):

    prediction = np.asarray(
        prediction,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Check number of classes
    # --------------------------------------------------------

    if prediction.ndim != 1:

        raise ValueError(
            f"Unexpected prediction shape: "
            f"{prediction.shape}"
        )


    if len(prediction) != 3:

        raise ValueError(
            f"Model returned "
            f"{len(prediction)} classes. "
            f"Expected exactly 3."
        )


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(prediction)
    ):

        raise ValueError(
            "Model returned NaN or infinite values."
        )


    # --------------------------------------------------------
    # If values don't look like probabilities,
    # treat them as logits and apply softmax.
    # --------------------------------------------------------

    total = float(
        np.sum(prediction)
    )


    if (
        np.any(prediction < 0)
        or total < 0.90
        or total > 1.10
    ):

        prediction = (
            get_tf().nn.softmax(
                prediction
            )
            .numpy()
        )


    else:

        # Normalize just in case
        prediction = (
            prediction / total
        )


    return prediction


# ============================================================
# PREDICT WITH ONE MODEL
# ============================================================

def predict_with_model(
    model,
    image,
    model_name
):

    print()
    print(
        "----------------------------------------------"
    )

    print(
        f"TESTING MODEL: {model_name}"
    )

    print(
        "----------------------------------------------"
    )


    # --------------------------------------------------------
    # Show model information
    # --------------------------------------------------------

    print(
        f"Model input shape: "
        f"{model.input_shape}"
    )


    print(
        f"Model output shape: "
        f"{model.output_shape}"
    )


    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image_array = (
        prepare_image_for_model(
            image,
            model
        )
    )


    print(
        f"Actual input sent to model: "
        f"{image_array.shape}"
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        raw_prediction = model(
            image_array,
            training=False
        ).numpy()

    except Exception as error:

        print()
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        print(
            f"PREDICTION ERROR: {model_name}"
        )

        print(
            f"ERROR: {error}"
        )

        traceback.print_exc()

        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        raise


    # --------------------------------------------------------
    # Show raw prediction
    # --------------------------------------------------------

    print(
        f"Raw prediction shape: "
        f"{raw_prediction.shape}"
    )


    print(
        f"Raw prediction: "
        f"{raw_prediction}"
    )


    # --------------------------------------------------------
    # Get first batch
    # --------------------------------------------------------

    prediction = raw_prediction[0]


    # --------------------------------------------------------
    # Convert to probabilities
    # --------------------------------------------------------

    prediction = (
        convert_to_probabilities(
            prediction
        )
    )


    print(
        f"Final probabilities: "
        f"{prediction}"
    )


    # --------------------------------------------------------
    # Best class
    # --------------------------------------------------------

    best_index = int(
        np.argmax(
            prediction
        )
    )


    class_name = (
        settings.CLASS_NAMES[
            best_index
        ]
    )


    confidence = float(
        prediction[
            best_index
        ]
    )


    print(
        f"RESULT: "
        f"{class_name.upper()} "
        f"({confidence * 100:.2f}%)"
    )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "model":
            model_name,

        "prediction":
            class_name,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "probabilities": {

            "paper":
                round(
                    float(
                        prediction[0]
                    ) * 100,
                    2
                ),

            "rock":
                round(
                    float(
                        prediction[1]
                    ) * 100,
                    2
                ),

            "scissors":
                round(
                    float(
                        prediction[2]
                    ) * 100,
                    2
                )

        },

        # Keep raw probability values
        # for ensemble averaging.

        "_prediction_array":
            prediction

    }


# ============================================================
# PREDICT API
# ============================================================

@require_POST
def predict(request):
    """
    Run the four-model ensemble sequentially to reduce peak memory use.
    Only one loaded Keras model is retained at a time.
    """
    print("\\n\\n================================================")
    print("         NEW PREDICTION REQUEST")
    print("================================================")

    image_file = request.FILES.get("image")
    if image_file is None:
        return JsonResponse(
            {"ok": False, "error": "No image received."},
            status=400,
        )

    print(f"Image name: {image_file.name}")
    print(f"Image size: {image_file.size} bytes")

    try:
        image = Image.open(image_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        print(f"IMAGE ERROR: {error}")
        return JsonResponse(
            {"ok": False, "error": "Invalid image."},
            status=400,
        )

    print(f"Received image dimensions: {image.size}")

    model_results = []
    all_predictions = []
    tf = None

    # Load, infer, and release each model before loading the next one.
    for model_path in settings.MODEL_PATHS:
        model = None
        model_name = model_path.name

        try:
            print("\\n==============================================")
            print(f"Loading model: {model_name}")
            print("==============================================")

            if not model_path.exists():
                raise FileNotFoundError(f"MODEL NOT FOUND: {model_path}")

            tf = get_tf()
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
            )

            result = predict_with_model(model, image, model_name)

            # Preserve the probability vector for ensemble averaging.
            all_predictions.append(
                np.asarray(
                    result.pop("_prediction_array"),
                    dtype=np.float32,
                ).copy()
            )
            model_results.append(result)

            print(f"Completed model: {model_name}")

        except Exception as error:
            print(f"MODEL FAILED: {model_name}: {error}")
            traceback.print_exc()
            return JsonResponse(
                {
                    "ok": False,
                    "error": (
                        f"{model_name} failed during prediction: "
                        f"{str(error)}"
                    ),
                },
                status=500,
            )

        finally:
            # Drop references before moving to the next model.
            if model is not None:
                del model

            if tf is not None:
                try:
                    tf.keras.backend.clear_session()
                except Exception:
                    traceback.print_exc()

            gc.collect()

    if len(model_results) != len(settings.MODEL_PATHS):
        return JsonResponse(
            {
                "ok": False,
                "error": (
                    f"Expected {len(settings.MODEL_PATHS)} model results, "
                    f"got {len(model_results)}."
                ),
            },
            status=500,
        )

    print("\\nALL MODELS PREDICTED SUCCESSFULLY")

    predictions_array = np.asarray(all_predictions, dtype=np.float32)
    print(f"Prediction array shape: {predictions_array.shape}")

    average_prediction = np.mean(predictions_array, axis=0)
    probability_sum = float(np.sum(average_prediction))
    if probability_sum > 0:
        average_prediction = average_prediction / probability_sum

    final_index = int(np.argmax(average_prediction))
    final_prediction = settings.CLASS_NAMES[final_index]
    final_confidence = float(average_prediction[final_index])

    final_probabilities = {
        "paper": round(float(average_prediction[0]) * 100, 2),
        "rock": round(float(average_prediction[1]) * 100, 2),
        "scissors": round(float(average_prediction[2]) * 100, 2),
    }

    votes = [result["prediction"] for result in model_results]
    vote_counts = {
        "paper": votes.count("paper"),
        "rock": votes.count("rock"),
        "scissors": votes.count("scissors"),
    }
    agreement = max(vote_counts.values())

    print("\\n================================================")
    print("             FINAL RPS RESULT")
    print("================================================")
    for result in model_results:
        print(
            f"{result['model']} -> {result['prediction'].upper()} "
            f"({result['confidence']}%)"
        )

    print(f"FINAL PREDICTION: {final_prediction.upper()}")
    print(f"FINAL CONFIDENCE: {final_confidence * 100:.2f}%")
    print(f"MODEL AGREEMENT: {agreement}/{len(model_results)}")
    print(f"FINAL PROBABILITIES: {final_probabilities}")
    print("================================================")

    return JsonResponse(
        {
            "ok": True,
            "prediction": final_prediction,
            "confidence": round(final_confidence * 100, 2),
            "probabilities": final_probabilities,
            "agreement": agreement,
            "votes": vote_counts,
            "models": model_results,
        }
    )
