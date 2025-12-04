FROM python:3.12-slim
WORKDIR /build

# Install system dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir gunicorn && \
    pip install --no-cache-dir\
    torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application
CMD ["gunicorn", "app.backend.app:fastapi_app", \
    "--bind", "0.0.0.0:10000", \
    "-k", "uvicorn.workers.UvicornH11Worker", \
    "--workers", "1", \
    "--log-level", "info", \
    "--access-logfile", "-", \
    "--error-logfile", "-"]