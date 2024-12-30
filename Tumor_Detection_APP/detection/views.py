from django.shortcuts import render, redirect
from .forms import MRIImageForm
from .models import MRIImage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained model
model = load_model('bestmodel.keras')

def predict_tumor(image_path):
    """Predict tumor presence in an MRI image."""
    image = load_img(image_path, target_size=(224, 224))  # Resize to match model input size
    image = img_to_array(image) / 255.0  # Normalize image
    image = image.reshape(1, 224, 224, 3)  # Add batch dimension
    prediction = model.predict(image)
    confidence = prediction[0][0]  # Assuming binary classification
    return ("Tumor Detected" if confidence > 0.5 else "No Tumor", confidence * 100)

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
    """Display prediction result."""
    mri_image = MRIImage.objects.get(pk=pk)
    return render(request, 'result.html', {'mri_image': mri_image})
