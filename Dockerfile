FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models/model.pkl /app/models/
COPY challenge/*.py /app/challenge/
COPY utils/*.py utils/

# Expose port
EXPOSE 8080

# Run API
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
