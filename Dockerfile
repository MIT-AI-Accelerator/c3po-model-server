FROM registry1.dso.mil/ironbank/opensource/python/python39:v3.9.16

WORKDIR /home/python

COPY --chown=python:python  ./ ./
COPY --chown=python:python  .cache/python-packages ./python-packages

ENV PYTHONPATH=/home/python/app:/home/python/python-packages:/home/python/tests
ENV ENVIRONMENT=staging

EXPOSE 8080

# ENTRYPOINT ["./scripts/init.sh"]

CMD ["sh", "-c", "cd /home/python && python -m uvicorn app.main:versioned_app --host 0.0.0.0 --port 8080"]
