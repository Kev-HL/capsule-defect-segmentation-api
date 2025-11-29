"""
FastAPI app for defect detection using a TFLite model.

Provided endpoints

- GET / : Render an HTML form (no results).
- POST /predict/ : REST API. Predict defect on an uploaded image; returns JSON.
- POST /upload/ : Upload image, run prediction, and return an HTML page with visualization and results.
- POST /random-sample/ : Run prediction on a random sample image and return an HTML page with visualization and results.

Design notes

- This application is a demonstration / portfolio app. For simplicity and safety during demo runs, inference is performed synchronously using a single global TFLite Interpreter instance protected by a threading.Lock to ensure thread-safety.
- The code intentionally makes a number of fixed assumptions about the model and runtime. If the model or deployment requirements change, the corresponding preprocessing, postprocessing and runtime setup should be updated and tested.

Assumptions:

    File system and assets
        Font used for drawing labels: ./fonts/OpenSans-Bold.ttf
        Static files served from: ./static
        Directories (./static/uploads, ./static/results, ./static/samples) are expected to be present/created by the deployment (Dockerfile or startup); added an exist_ok mkdir as safeguard.

    Upload / input constraints
        Uploaded images are expected to be valid PNG images (this matches the local MVTec AD dataset used for development).
        Maximum accepted upload size: 5 MB.

    Runtime / model
        Uses tflite-runtime Interpreter for model inference (Interpreter from tflite_runtime.interpreter).
        TFLite model file path: ./final_model.tflite
        Single Interpreter instance is created at startup and reused for all requests (protected by a threading.Lock).

    Model I/O (these are the exact assumptions used by the code)
        Expected input tensor: shape (1, 512, 512, 3), dtype float32, pixel value range [0, 255] (model handles internally normalization to [0, 1]).
        Expected output[0]: segmentation mask of shape (1, 512, 512, 1), dtype float32, values in [0, 1] (probability map).
        Expected output[1]: class probabilities of shape (1, 6), dtype float32 (softmax-like probabilities).
"""

# IMPORTS

# Standard library imports
import io
import logging
import os
import random
import time
import threading

# Third-party imports
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from tflite_runtime.interpreter import Interpreter
# from ai_edge_litert.interpreter import Interpreter

# Auxiliary imports (Dockerfile sets CWD to /app)
from aux import preprocess_image, inference, save_image, draw_prediction, encode_mask_to_base64, delete_files_later

# START TIME LOGGING
import time
app_start = time.perf_counter()

# CONFIGURATION AND CONSTANTS

# Path to TFLite model file
MODEL_PATH = './final_model.tflite'

# Number of threads for TFLite interpreter
NUM_THREADS = 4

# Jinja2 templates directory
TEMPLATES = Jinja2Templates(directory='templates')

# Max file size for uploads (5 MB)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


# MAIN APPLICATION


# Set up logging to show INFO level and above messages
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Mount static files directory for serving images and other assets
# App will raise errors if folders do not exist
# Directory creation is handled by the Dockerfile
os.makedirs('static', exist_ok=True)
app.mount('/static', StaticFiles(directory='static'), name='static')

# Load model, set up interpreter and get input/output details
try:
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
except:
    logging.warning(f'num_threads={NUM_THREADS} not supported, falling back to single-threaded interpreter.')
    interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logging.info('TF Lite input details: %s \n', input_details)
logging.info('TF Lite output details: %s \n', output_details)

# Create a threading lock for the interpreter to ensure thread-safety
interpreter_lock = threading.Lock()

# Inference context to be passed to inference function
inference_ctx = {
    'interpreter_lock': interpreter_lock,
    'interpreter': interpreter,
    'input_details': input_details,
    'output_details': output_details,
}

# Startup time measurement
@app.on_event('startup')
async def report_startup_time():
    startup_time = (time.perf_counter() - app_start) * 1000  # in milliseconds
    logging.info(f'App startup time: {startup_time:.2f} ms \n')

# Root endpoint to render the HTML form
@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    # Render the HTML form with empty image URLs and no result
    return TEMPLATES.TemplateResponse(
        'index.html',
        {
            'request': request,
            'result': None,
            'orig_img_url': None,
            'vis_img_url': None,
        }
    )

# Endpoint to handle image prediction (API)
@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        # Check if the uploaded file is a PNG image
        if file.content_type != 'image/png':
            return JSONResponse(status_code=400, content={'error': 'Only PNG images are supported.'})

        # Read the image
        image_bytes = await file.read()

        # Check if the file size exceeds the maximum limit
        if len(image_bytes) > MAX_FILE_SIZE:
            return JSONResponse(status_code=400, content={'error': 'File size exceeds the maximum limit of 5 MB.'})
        
        # Check if the image is a valid PNG (not just a file with .png extension)
        try:
            img_check = Image.open(io.BytesIO(image_bytes))
            if img_check.format != 'PNG':
                raise ValueError('Not a PNG')
        except (UnidentifiedImageError, ValueError):
            return JSONResponse(status_code=400, content={'error': 'Invalid image file.'})

        # Preprocess the image
        img = preprocess_image(image_bytes)

        # Run inference on the preprocessed image
        class_id, class_name, mask = inference(img, inference_ctx)

        # Encode mask to base64
        mask64 = encode_mask_to_base64(mask)

        # Return the prediction results as JSON
        return {
            'class_id': class_id,
            'class_name': class_name,
            'mask64_PNG_L': mask64,
        }
    except Exception as e:
        logging.exception(f'Error during prediction: {e}')
        return JSONResponse(status_code=500, content={'error': 'Model inference failed.'})

# Endpoint to handle image upload and prediction with visualization
@app.post('/upload/', response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    try:
        # Check if the uploaded file is a PNG image
        if file.content_type != 'image/png':
            result = {'error': 'Only PNG images are supported.'}
            return TEMPLATES.TemplateResponse('index.html', {'request': request, 'result': result})

        # Read the uploaded image
        image_bytes = await file.read()

        # Check if the file size exceeds the maximum limit
        if len(image_bytes) > MAX_FILE_SIZE:
            return TEMPLATES.TemplateResponse('index.html', {'request': request, 'result': {'error': 'File too large (max 5MB).'}})

        # Check if the image is a valid PNG (not just a file with .png extension)
        try:
            img_check = Image.open(io.BytesIO(image_bytes))
            if img_check.format != 'PNG':
                raise ValueError('Not a PNG')
        except (UnidentifiedImageError, ValueError):
            return TEMPLATES.TemplateResponse('index.html', {'request': request, 'result': {'error': 'Invalid image file.'}})

        # Save the preprocessed image
        preproc_filename, preproc_path = save_image(image_bytes)
    
        # Preprocess the image
        img = preprocess_image(image_bytes)

        # Run inference on the preprocessed image
        class_id, class_name, mask = inference(img, inference_ctx)

        # Overlay mask and draw class name on the preprocessed image for display
        pred_filename, pred_path = draw_prediction(preproc_path, mask, class_name)

        # Encode mask to base64
        mask64 = encode_mask_to_base64(mask)

        # Prepare the result to be displayed in the HTML template
        result = {
            'class_id': class_id,
            'class_name': class_name,
            'mask64_PNG_L': mask64,
        }

        # Schedule deletion of both images after 10 seconds
        if background_tasks is not None:
            background_tasks.add_task(delete_files_later, [preproc_path, pred_path], delay=10)

        # Render the HTML template with the result and image URLs
        return TEMPLATES.TemplateResponse(
            'index.html',
            {
                'request': request,
                'result': result,
                'preproc_img_url': f'/static/uploads/{preproc_filename}',
                'pred_img_url': f'/static/results/{pred_filename}',
            }
        )
    except Exception as e:
        logging.exception(f'Error during prediction: {e}')
        return TEMPLATES.TemplateResponse('index.html', {'request': request, 'result': {'error': 'Model inference failed.'}})

# Endpoint to handle random image (from samples) prediction with visualization
@app.post('/random-sample/', response_class=HTMLResponse)
async def random_sample(request: Request, background_tasks: BackgroundTasks = None):
    try:
        # Check if the samples directory exists and contains PNG files
        samples_dir = 'static/samples'
        sample_files = [f for f in os.listdir(samples_dir) if f.lower().endswith('.png')]
        if not sample_files:
            result = {'error': 'No sample images available.'}
            return TEMPLATES.TemplateResponse('index.html', {'request': request, 'result': result})
        
        # Randomly select a sample image and read it
        chosen_file = random.choice(sample_files)
        with open(os.path.join(samples_dir, chosen_file), 'rb') as f:
            image_bytes = f.read()
        
        # Save preprocessed image
        preproc_filename, preproc_path = save_image(image_bytes)
        
        # Preprocess the image
        img = preprocess_image(image_bytes)

        # Run inference on the preprocessed image
        class_id, class_name, mask = inference(img, inference_ctx)

        # Overlay mask and draw class name on the preprocessed image for display
        pred_filename, pred_path = draw_prediction(preproc_path, mask, class_name)

        # Encode mask to base64
        mask64 = encode_mask_to_base64(mask)

        # Prepare the result to be displayed in the HTML template
        result = {
            'class_id': class_id,
            'class_name': class_name,
            'mask64_PNG_L': mask64,
        }

        # Schedule deletion of both images after 10 seconds
        if background_tasks is not None:
            background_tasks.add_task(delete_files_later, [preproc_path, pred_path], delay=10)

        # Render the HTML template with the result and image URLs
        return TEMPLATES.TemplateResponse(
            'index.html',
            {
                'request': request,
                'result': result,
                'preproc_img_url': f'/static/uploads/{preproc_filename}',
                'pred_img_url': f'/static/results/{pred_filename}',
            }
        )
    except Exception as e:
        logging.exception(f'Error during prediction: {e}')
        return TEMPLATES.TemplateResponse('index.html', {'request': request, 'result': {'error': 'Model inference failed.'}})