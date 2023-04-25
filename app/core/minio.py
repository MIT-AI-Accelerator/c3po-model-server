from minio import Minio
from .config import settings

def build_client():
    return Minio(
            settings.minio_endpoint_url,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
        )
