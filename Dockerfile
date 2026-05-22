FROM python:3.11-slim

# Hugging Face requires running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install requirements
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files with proper permissions
COPY --chown=user . /app

EXPOSE 7860

# Run Flask via Gunicorn with 1 worker and threads to save RAM on free tiers
CMD ["gunicorn", "-w", "1", "--threads", "2", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120"]
