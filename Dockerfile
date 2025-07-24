# Specify the platform as required by the hackathon [cite: 57]
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model and the source code into the container
COPY ./models/ /app/models/
COPY ./src/ /app/src/

# Define the command that will run your solution
# This main.py script must handle the I/O from /app/input and /app/output
CMD [ "python", "/app/src/main.py" ]