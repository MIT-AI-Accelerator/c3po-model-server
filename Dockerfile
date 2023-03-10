FROM registry1.dso.mil/ironbank/opensource/python/python39:v3.9.16

WORKDIR /home/python

COPY --chown=python:python  ./ ./
COPY --chown=python:python  .cache/python-packages ./python-packages

ENV PYTHONPATH=/home/python/app:/home/python/python-packages:/home/python/tests

EXPOSE 8080

CMD ["sh", "-c", "cd /home/python && ENVIRONMENT='staging' python3 -m uvicorn app.api:app --host 0.0.0.0 --port 8080"]
