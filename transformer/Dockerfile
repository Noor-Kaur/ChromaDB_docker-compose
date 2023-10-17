FROM python:3.9

WORKDIR /transformer

COPY ./requirements.txt /transformer/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /transformer/requirements.txt

COPY . /transformer

CMD ["uvicorn", "bge_embedding_function:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "5000"]