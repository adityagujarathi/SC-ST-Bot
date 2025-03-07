FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Java for tabula-py
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
