FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code

RUN pip install -U -r /code/requirements.txt

COPY . /code

CMD ["uvicorn", "chroma_service:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]