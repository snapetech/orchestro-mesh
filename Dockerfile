FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir -e .
EXPOSE 8765
CMD ["uvicorn", "orchestro_mesh.gateway:app", "--host", "0.0.0.0", "--port", "8765"]
