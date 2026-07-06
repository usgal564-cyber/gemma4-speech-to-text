FROM python:3.12-slim

# cpu (default) or cu128 for CUDA builds:
#   docker build --build-arg TORCH_VARIANT=cu128 .
ARG TORCH_VARIANT=cpu

# git: transformers is pinned to a git commit in requirements.txt
# ffmpeg: decode mp3/webm/ogg uploads and mic recordings
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch first from the variant-specific index so requirements.txt
# (torch==2.8.0) is already satisfied and pip does not pull CUDA wheels.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    torch==2.8.0 torchvision==0.23.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio

COPY app.py .

ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
