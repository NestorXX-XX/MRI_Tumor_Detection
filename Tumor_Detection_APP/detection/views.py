from django.shortcuts import render, redirect
from .forms import MRIImageForm
from .models import MRIImage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import os
from django.conf import settings  # For MEDIA_ROOT and MEDIA_URL
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


# Load the pre-trained model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bestmodel.keras')
model = load_model(model_path)

def preprocess_image(path):
    """Preprocess a single image for prediction."""
    img = image.load_img(path, target_size=(224, 224))  # Resize image
    input_arr = img_to_array(img)  # Convert image to array
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    input_arr = preprocess_input(input_arr)  # Apply preprocessing function
    return input_arr

def get_prediction(path):
    """Get prediction and raw probability for a given image."""
    input_arr = preprocess_image(path)
    prob = model.predict(input_arr)[0][0]  # Extract scalar float
    print(f"Raw probability: {prob}")  # Debugging
    pred = 1 if prob >= 0.5 else 0
    return pred, prob

def log_misclassification(image_path, prob, pred):
    """Log misclassified images for debugging."""
    if pred == 0 and prob > 0.5:
        print(f"Misclassified as No Tumor: {image_path}, Probability: {prob}")
    elif pred == 1 and prob < 0.5:
        print(f"Misclassified as Tumor: {image_path}, Probability: {prob}")

def predict_tumor(image_path):
    """Predict tumor presence and confidence for an MRI image."""
    pred, prob = get_prediction(image_path)
    log_misclassification(image_path, prob, pred)  # Log misclassification

    # Ensure confidence is calculated properly
    if pred == 0:
        result = "No Tumor"
        confidence = (1 - prob) * 100  # Confidence for healthy
    else:
        result = "Tumor Detected"
        confidence = prob * 100  # Confidence for tumor

    confidence = round(confidence, 2)  # Limit to two decimal places
    print(f"Prediction: {result}, Raw Probability: {prob}, Confidence: {confidence}%")
    return result, confidence

def save_plot(image_array, save_path):
    """Save the plot of the image array to a file."""
    plt.imshow(image_array)
    plt.axis('off')  # Turn off axes for a cleaner display
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the figure to release resources

def upload_image(request):
    """Handle image upload and tumor prediction."""
    if request.method == 'POST':
        form = MRIImageForm(request.POST, request.FILES)
        if form.is_valid():
            mri_image = form.save()  # Save uploaded image
            # Predict tumor
            prediction, confidence = predict_tumor(mri_image.image.path)
            mri_image.prediction = prediction
            mri_image.confidence = confidence
            mri_image.save()  # Save the prediction and confidence
            return redirect('result', pk=mri_image.pk)
    else:
        form = MRIImageForm()
    return render(request, 'upload_image.html', {'form': form})

def result(request, pk):
    """Display prediction result and input image."""
    mri_image = MRIImage.objects.get(pk=pk)
    img = load_img(mri_image.image.path, target_size=(224, 224))
    input_arr = img_to_array(img) / 255.0

    # Save the input image plot to the media directory
    plot_save_path = os.path.join(settings.MEDIA_ROOT, 'plots', f'result_{pk}.png')
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)  # Ensure the directory exists
    save_plot(input_arr, plot_save_path)

    # Get the relative URL for the media file
    plot_media_url = os.path.join(settings.MEDIA_URL, 'plots', f'result_{pk}.png')
    return render(request, 'result.html', {
        'mri_image': mri_image,
        'result_image': plot_media_url,  # Pass the relative URL for the media file
    })
