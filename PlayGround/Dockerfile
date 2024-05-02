FROM python:3.9

# Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the application files
COPY --chown=user . $HOME/app

# Copy the requirements file and install dependencies
COPY requirements.txt $HOME/app/requirements.txt
RUN pip install --user -r requirements.txt

# Expose the port and set the command to run the application
CMD ["chainlit", "run", "app.py", "--port", "7860"]

