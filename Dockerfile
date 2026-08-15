FROM python:3.11-slim

EXPOSE 5000
# EXPOSE 8501

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu
    
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
