import io
import pickle
from typing import Any, Optional, Union
from .config import settings
from .logging import logger
from fastapi import UploadFile, HTTPException
from minio.error import InvalidResponseError
from minio import Minio
from pydantic import UUID4

def build_client():

    if not settings.minio_region:
        return Minio(
                settings.minio_endpoint_url,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure
            )

    return Minio(
            settings.minio_endpoint_url,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            region=settings.minio_region
        )

def upload_file_to_minio(file: UploadFile, id: UUID4, s3: Minio) -> bool:
    output_filename = f"{id}"

    # calculate the size of the file
    file.file.seek(0,2) # move the cursor to the end of the file
    file_size = file.file.tell() # get the position of EOF
    file.file.seek(0) # move the cursor to the start of the file

    try:
        s3.put_object(
            bucket_name=settings.minio_bucket_name,
            object_name=output_filename,
            data=file.file,
            length=file_size,
            content_type='application/octet-stream'
        )
    except InvalidResponseError as e:
        logger.error(f"Failed to upload file to Minio: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    return True

def pickle_and_upload_object_to_minio(object: Any, id: UUID4, s3: Minio) -> bool:

    # Create an in-memory file object
    file_obj = io.BytesIO()

    # Dump the object to the file object
    pickle.dump(object, file_obj)

    # Move the file cursor to the beginning of the file
    file_obj.seek(0)

    # utilize id from above to upload file to minio
    upload_file_to_minio(UploadFile(file_obj),
                            id, s3)

    return True

def download_file_from_minio(id: Union[UUID4, str], s3: Minio, filename: Optional[str] = None) -> io.BytesIO:
    output_filename = f"{str(id)}"

    try:
        data = s3.get_object(
            bucket_name=settings.minio_bucket_name,
            object_name=output_filename
        )

        file_obj = None
        write_to_memory = not filename
        if write_to_memory:
            file_obj = io.BytesIO()
        else:
            file_obj = open(filename, 'wb')

        for d in data.stream(32*1024):
            file_obj.write(d)

        file_obj.seek(0)

        if not write_to_memory:
            file_obj.close()

        return file_obj
    except InvalidResponseError as e:
        logger.error(f"Failed to download file from Minio: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e
    finally:
        data.close()
        data.release_conn()

def download_pickled_object_from_minio(id: UUID4, s3: Minio) -> Any:
    file_obj = download_file_from_minio(id, s3)
    output = pickle.load(file_obj)

    file_obj.close()
    return output
