# Use an official Python runtime as a parent image
FROM python:3.6

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN git clone https://github.com/fizyr/keras-retinanet.git
RUN pip install keras-retinanet/. --user
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install opencv-python==3.4.1.15
RUN pip install Pillow
RUN mv resnet50_coco_best_v2.1.0.h5 keras-retinanet/snapshots/resnet50_coco_best_v2.1.0.h5

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME AvitoDetector

# Run app.py when the container launches
# CMD ["python", "avito_detect.py"]
ENTRYPOINT ["python", "avito_detect.py"]
