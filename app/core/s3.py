import io
import pickle
import boto3
from typing import Any, Optional, Union
from .config import settings
from .logging import logger
from fastapi import UploadFile, HTTPException
from pydantic import UUID4
from mypy_boto3_s3.client import S3Client
from botocore.response import StreamingBody


def build_client() -> S3Client:

    if not settings.s3_region:
        return boto3.client(
            's3',
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            use_ssl=settings.s3_secure
        )

    return boto3.client(
        's3',
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        use_ssl=settings.s3_secure,
        region_name=settings.s3_region
    )

def upload_file_to_s3(file: UploadFile, id: UUID4, s3: S3Client) -> bool:
    output_filename = f"{id}"

    # calculate the size of the file
    file.file.seek(0,2) # move the cursor to the end of the file
    file_size = file.file.tell() # get the position of EOF
    file.file.seek(0) # move the cursor to the start of the file

    try:
        s3.put_object(
            Bucket=settings.s3_bucket_name,
            Key=output_filename,
            Body=file.file,
            ContentLength=file_size,
            ContentType='application/octet-stream'
        )
    except Exception as e:
        logger.error(f"Failed to upload file to S3: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    return True

def pickle_and_upload_object_to_s3(object: Any, id: UUID4, s3: S3Client) -> bool:

    # Create an in-memory file object
    file_obj = io.BytesIO()

    # Dump the object to the file object
    pickle.dump(object, file_obj)

    # Move the file cursor to the beginning of the file
    file_obj.seek(0)

    # utilize id from above to upload file to s3
    upload_file_to_s3(UploadFile(file_obj),
                      id, s3)

    return True

def download_file_from_s3(id: Union[UUID4, str], s3: S3Client, filename: Optional[str] = None) -> io.BytesIO:
    output_filename = f"{str(id)}"
    data = None

    try:
        response = s3.get_object(
            Bucket=settings.s3_bucket_name,
            Key=output_filename
        )
        data: StreamingBody = response['Body']

        file_obj = None
        write_to_memory = not filename
        if write_to_memory:
            file_obj = io.BytesIO()
        else:
            file_obj = open(filename, 'wb')

        for d in data.iter_chunks(chunk_size=32*1024):
            file_obj.write(d)

        file_obj.seek(0)

        if not write_to_memory:
            file_obj.close()

        return file_obj
    except Exception as e:
        logger.error(f"Failed to download file from S3: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e
    finally:
        if data is not None:
            data.close()

def download_pickled_object_from_s3(id: UUID4, s3: S3Client) -> Any:
    file_obj = download_file_from_s3(id, s3)
    output = pickle.load(file_obj)

    file_obj.close()
    return output

def list_s3_objects(s3: S3Client) -> Any:
    """Lists all objects in the specified bucket."""

    try:
        logger.info(f'Listing objects in bucket {settings.s3_bucket_name}')
        response = s3.list_objects_v2(Bucket=settings.s3_bucket_name)
        logger.info("S3 objects:")
        for obj in response['Contents']:
            logger.info(f"Key: {obj['Key']}, Last modified: {obj['LastModified']}, Size (B): {obj['Size']}")
    except: # pylint: disable=bare-except
        logger.warning(f"unable to list s3 objects for {settings.s3_bucket_name}")
