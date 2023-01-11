FROM registry1.dso.mil/ironbank/opensource/tensorflow/tensorflow-2.5.1:2.5.1

WORKDIR /home/python

COPY --chown=tensorflow:tensorflow  ./ ./
COPY --chown=tensorflow:tensorflow  .cache/python-packages ./python-packages

ENV PYTHONPATH=/home/python/app:/home/python/python-packages

CMD ["sh", "-c", "cd /home/python && python -m puckboard_solver.cli.entrypoint run-webapp --host 0.0.0.0 --port 8080"]
