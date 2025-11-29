# Capsule Defect Detection and Segmentation with ConvNeXt+U-Net and FastAPI
# Use slim Python image for smaller size
FROM python:3.9.23-slim-bookworm

# Basic ownership labels
LABEL maintainer="Kev-HL (GitHub)"
LABEL org.opencontainers.image.source="https://github.com/Kev-HL/capsule-defect-segmentation-api"

# Set working directory
WORKDIR /app

# Create a non-root user and group (appuser)
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser

# Update system packages and clean up
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt for API dependencies
COPY app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow Lite runtime from local wheel file
# Remove or comment if using a different interpreter (tflite-runtime or ai-edge-litert)
COPY --chown=appuser:appuser app/tflite_runtime-2.19.0-cp39-cp39-linux_x86_64.whl .
RUN pip install --no-cache-dir ./tflite_runtime-2.19.0-cp39-cp39-linux_x86_64.whl

# Clean up
RUN rm tflite_runtime-2.19.0-cp39-cp39-linux_x86_64.whl && \
    find /usr/local/lib/python3.9/ -type d -name '__pycache__' -prune -exec rm -rf {} + && \
    rm -rf /usr/share/doc /usr/share/man /usr/share/info /usr/share/locale/*

# Copy app code (FastAPI app)
COPY --chown=appuser:appuser app/main.py .

# Copy aux code (functions for FastAPI app)
COPY --chown=appuser:appuser app/aux.py .

# Copy model file
COPY --chown=appuser:appuser models/final_model/final_model.tflite .

# Copy HTML templates
RUN mkdir -p templates && chown -R appuser:appuser templates
COPY --chown=appuser:appuser app/templates/ templates/

# Create static directories for uploads, results and samples
RUN mkdir -p static/uploads static/results static/samples && chown -R appuser:appuser static

# Copy sample images
COPY --chown=appuser:appuser app/samples/ static/samples/

# Copy font file (and license) for text rendering on images
RUN mkdir -p fonts && chown -R appuser:appuser fonts
COPY --chown=appuser:appuser app/fonts/OpenSans-Bold.ttf fonts
COPY --chown=appuser:appuser app/fonts/OFL.txt fonts

# Set permissions for static files
RUN chmod -R 777 static

# Switch to non-root user
USER appuser

# Expose port (FastAPI default)
EXPOSE 8000

# Set environment variables
# Disable buffering for easier logging (immediate output)
ENV PYTHONUNBUFFERED=1

# Start FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
