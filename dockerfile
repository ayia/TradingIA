# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files
COPY Api.py /app/
COPY train_lstm.py /app/
COPY requirements.txt /app/
COPY Trained.models /app/Trained.models/


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the Flask app runs on
EXPOSE 8073

# Set the command to run the API
CMD ["python", "Api.py"]
