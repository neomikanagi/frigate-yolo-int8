# Build environment for every package: Ultralytics export, OpenVINO + NNCF, RKNN Toolkit 2, ONNX Runtime.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
ARG ULTRALYTICS_VERSION
RUN touch /tmp/constraints.txt \
    && if [ -n "$ULTRALYTICS_VERSION" ]; then \
         printf 'ultralytics==%s\n' "$ULTRALYTICS_VERSION" > /tmp/constraints.txt; \
       fi \
    && pip install --no-cache-dir --root-user-action=ignore \
         -r requirements.txt -c /tmp/constraints.txt

COPY yolo_common.py build_onnx.py build_openvino.py build_rknn.py ./
