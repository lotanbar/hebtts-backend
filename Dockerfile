FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV AUDIOCRAFT_CACHE_DIR=/app/.audiocraft_cache
ENV HF_HOME=/app/.hf_cache
ENV TORCH_HOME=/app/.torch_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    git+https://github.com/lhotse-speech/lhotse.git gdown

COPY . .

# Download models with CORRECT settings
WORKDIR /app/HebTTSLM
RUN gdown 11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv -O checkpoint.pt

RUN python3 -c "from audiocraft.models import MultiBandDiffusion; MultiBandDiffusion.get_mbd_24khz(bw=6.0)"
RUN python3 -c "from encodec import EncodecModel; EncodecModel.encodec_model_24khz()"

WORKDIR /app
ENV PYTHONPATH=/app/HebTTSLM:$PYTHONPATH
CMD ["python", "-u", "handler.py"]