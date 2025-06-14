# Use a specific, stable Python version
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy your requirements and source files
COPY requirements.txt .
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "main.py"]
