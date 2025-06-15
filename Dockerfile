# Use a specific, stable Python version
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Copy your requirements and source files
COPY . /app

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "main.py"]
