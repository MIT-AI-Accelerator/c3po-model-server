FROM registry1.dso.mil/ironbank/opensource/python:v3.11.4

WORKDIR /home/python

COPY --chown=python:python  ./ ./
COPY --chown=python:python  .cache/python-packages ./python-packages

ENV PYTHONPATH=/home/python/app:/home/python/python-packages:/home/python/tests
ENV ENVIRONMENT=staging

EXPOSE 8080

ENTRYPOINT ["./scripts/entrypoint.sh"]

CMD ["python", "-m", "uvicorn", "app.main:versioned_app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
