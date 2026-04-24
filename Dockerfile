FROM registry1.dso.mil/ironbank/opensource/python:v3.13

WORKDIR /home/python

COPY --chown=python:python  ./ ./

ENV PYTHONPATH=/home/python/app:/home/python/.cache/python-packages:/home/python/tests
ENV ENVIRONMENT=staging

EXPOSE 8080

ENTRYPOINT ["./scripts/entrypoint.sh"]

CMD ["python", "-m", "uvicorn", "app.main:versioned_app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
