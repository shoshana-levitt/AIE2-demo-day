FROM python:3.9
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY requirements.txt $HOME/app/requirements.txt
RUN pip install --user -r requirements.txt
CMD ["chainlit", "run", "app.py", "--port", "7860"]
