FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models
WORKDIR /srv

# CPU-only torch first (the default PyPI wheel bundles CUDA and is several GB); no GPU is required.
COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/documents ./data/documents
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
ENTRYPOINT ["./docker-entrypoint.sh"]
