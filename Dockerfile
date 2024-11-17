# Base image for Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the API port
EXPOSE 5000

# Set the default command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
