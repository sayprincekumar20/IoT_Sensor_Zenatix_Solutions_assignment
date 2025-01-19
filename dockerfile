# Base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy the Python script and other required files into the container
COPY IoT_Sensor_Script.py /app/
COPY IoT_Sensor_Data.csv /app/
COPY requirements.txt /app/


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port if the script needs one (adjust as necessary)
EXPOSE 6000

# Set the entry point to run your Python script
CMD ["python", "IoT_Sensor_Script.py"]
