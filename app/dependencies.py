from collections.abc import Generator
import httpx
from .db.session import SessionLocal
from .aimodels.lstm_stress_classifier.ai_service.inference.inference_model import LstmStressClassifierModel
from .core.s3 import build_client
from mypy_boto3_s3.client import S3Client



# see here for config suggestions: https://stackoverflow.com/questions/74184899/is-having-a-concurrent-futures-threadpoolexecutor-call-dangerous-in-a-fastapi-en/74239367#74239367
limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
timeout = httpx.Timeout(10.0, read=20.0)
httpx_client = httpx.AsyncClient(limits=limits, timeout=timeout)

# ********use for DI********
lstm_stress_classifier_model = LstmStressClassifierModel()
def get_lstm_stress_classifier_model():
    return lstm_stress_classifier_model

# ********use for DB initialization*****
def get_db() -> Generator:
    db = None
    try:
        db = SessionLocal()
        yield db
    finally:
        if db is not None:
            db.close()

# ********use for s3 initialization*****
def get_s3() -> S3Client:
    return build_client()
