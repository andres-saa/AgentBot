FROM python:3.12 AS base
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS dev
ENV PYTHONUNBUFFERED=1
COPY ./app /code/
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","80","--reload"]

FROM base AS prod
RUN pip install --no-cache-dir gunicorn uvicorn[standard]
COPY ./app /code/
CMD ["gunicorn","-k","uvicorn.workers.UvicornWorker","-w","4","-t","1","-b","0.0.0.0:80","main:app"]
