import io
import pickle
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4
import pytest
from botocore.exceptions import BotoCoreError, EndpointConnectionError
from fastapi import HTTPException, UploadFile
from mypy_boto3_s3.client import S3Client

from app.core.s3 import (
    build_client,
    upload_file_to_s3,
    pickle_and_upload_object_to_s3,
    download_file_from_s3,
    download_pickled_object_from_s3,
    list_s3_objects
)


def test_build_client_without_region():
    """Test building S3 client without region"""
    with patch('app.core.s3.boto3.client') as mock_boto3_client:
        with patch('app.core.s3.settings') as mock_settings:
            mock_settings.s3_region = None
            mock_settings.s3_endpoint_url = "http://localhost:9000"
            mock_settings.s3_access_key = "test_key"
            mock_settings.s3_secret_key = "test_secret"
            mock_settings.s3_secure = False
            
            build_client()
            
            mock_boto3_client.assert_called_once_with(
                's3',
                endpoint_url="http://localhost:9000",
                aws_access_key_id="test_key",
                aws_secret_access_key="test_secret",
                use_ssl=False
            )


def test_build_client_with_region():
    """Test building S3 client with region"""
    with patch('app.core.s3.boto3.client') as mock_boto3_client:
        with patch('app.core.s3.settings') as mock_settings:
            mock_settings.s3_region = "us-east-1"
            mock_settings.s3_endpoint_url = "https://s3.amazonaws.com"
            mock_settings.s3_access_key = "test_key"
            mock_settings.s3_secret_key = "test_secret"
            mock_settings.s3_secure = True
            
            build_client()
            
            mock_boto3_client.assert_called_once_with(
                's3',
                endpoint_url="https://s3.amazonaws.com",
                aws_access_key_id="test_key",
                aws_secret_access_key="test_secret",
                use_ssl=True,
                region_name="us-east-1"
            )


def test_build_client_boto_error():
    """Test that build_client handles exceptions during client creation"""
    with patch('app.core.s3.boto3.client') as mock_boto3_client:
        with patch('app.core.s3.settings') as mock_settings:
            mock_settings.s3_region = None
            mock_settings.s3_endpoint_url = "http://localhost:9000"
            mock_settings.s3_access_key = "test_key"
            mock_settings.s3_secret_key = "test_secret"
            mock_settings.s3_secure = True
            
            mock_boto3_client.side_effect = EndpointConnectionError(endpoint_url="http://localhost:9000")
            
            with pytest.raises(Exception):
                build_client()


def test_upload_file_to_s3_success():
    """Test successful file upload to S3"""
    test_id = uuid4()
    mock_s3 = MagicMock(spec=S3Client)
    
    file_content = b"test content"
    file_obj = io.BytesIO(file_content)
    upload_file = UploadFile(file_obj, filename="test.txt")
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        result = upload_file_to_s3(upload_file, test_id, mock_s3)
        
        assert result is True
        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs['Bucket'] == 'test-bucket'
        assert call_kwargs['Key'] == str(test_id)
        assert call_kwargs['ContentType'] == 'application/octet-stream'


def test_upload_file_to_s3_boto_error():
    """Test upload_file_to_s3 handles BotoCoreError"""
    test_id = uuid4()
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.put_object.side_effect = BotoCoreError()
    
    file_obj = io.BytesIO(b"test")
    upload_file = UploadFile(file_obj, filename="test.txt")
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        with pytest.raises(HTTPException) as exc_info:
            upload_file_to_s3(upload_file, test_id, mock_s3)
        
        assert exc_info.value.status_code == 500


def test_pickle_and_upload_object_to_s3():
    """Test pickling and uploading an object to S3"""
    test_id = uuid4()
    test_object = {"key": "value", "number": 42}
    mock_s3 = MagicMock(spec=S3Client)
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        result = pickle_and_upload_object_to_s3(test_object, test_id, mock_s3)
        
        assert result is True
        mock_s3.put_object.assert_called_once()


def test_download_file_from_s3_to_memory():
    """Test downloading file from S3 to memory"""
    test_id = uuid4()
    test_content = b"downloaded content"
    
    mock_streaming_body = MagicMock()
    mock_streaming_body.iter_chunks.return_value = [test_content]
    mock_streaming_body.close = MagicMock()
    
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.get_object.return_value = {'Body': mock_streaming_body}
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        result = download_file_from_s3(test_id, mock_s3)
        
        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read() == test_content
        mock_streaming_body.close.assert_called_once()


def test_download_file_from_s3_to_file(tmp_path):
    """Test downloading file from S3 to disk"""
    test_id = uuid4()
    test_content = b"file content"
    test_file = tmp_path / "test_download.txt"
    
    mock_streaming_body = MagicMock()
    mock_streaming_body.iter_chunks.return_value = [test_content]
    mock_streaming_body.close = MagicMock()
    
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.get_object.return_value = {'Body': mock_streaming_body}
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        download_file_from_s3(test_id, mock_s3, filename=str(test_file))
        
        assert test_file.exists()
        assert test_file.read_bytes() == test_content


def test_download_file_from_s3_boto_error():
    """Test download_file_from_s3 handles BotoCoreError"""
    test_id = uuid4()
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.get_object.side_effect = BotoCoreError()
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        with pytest.raises(HTTPException) as exc_info:
            download_file_from_s3(test_id, mock_s3)
        
        assert exc_info.value.status_code == 500


def test_download_file_from_s3_with_string_id():
    """Test downloading file with string ID instead of UUID"""
    test_id = "test-string-id"
    test_content = b"content"
    
    mock_streaming_body = MagicMock()
    mock_streaming_body.iter_chunks.return_value = [test_content]
    mock_streaming_body.close = MagicMock()
    
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.get_object.return_value = {'Body': mock_streaming_body}
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        result = download_file_from_s3(test_id, mock_s3)
        
        assert isinstance(result, io.BytesIO)
        mock_s3.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='test-string-id'
        )


def test_download_pickled_object_from_s3():
    """Test downloading and unpickling an object from S3"""
    test_id = uuid4()
    test_object = {"key": "value", "list": [1, 2, 3]}
    pickled_content = pickle.dumps(test_object)
    
    mock_streaming_body = MagicMock()
    mock_streaming_body.iter_chunks.return_value = [pickled_content]
    mock_streaming_body.close = MagicMock()
    
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.get_object.return_value = {'Body': mock_streaming_body}
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        result = download_pickled_object_from_s3(test_id, mock_s3)
        
        assert result == test_object


def test_list_s3_objects_success():
    """Test listing S3 objects successfully"""
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.list_objects_v2.return_value = {
        'Contents': [
            {'Key': 'file1.txt', 'LastModified': '2024-01-01', 'Size': 1024},
            {'Key': 'file2.txt', 'LastModified': '2024-01-02', 'Size': 2048}
        ]
    }
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        list_s3_objects(mock_s3)
        
        mock_s3.list_objects_v2.assert_called_once_with(Bucket='test-bucket')


def test_list_s3_objects_boto_error():
    """Test list_s3_objects handles BotoCoreError gracefully"""
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.list_objects_v2.side_effect = BotoCoreError()
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        list_s3_objects(mock_s3)
        
        mock_s3.list_objects_v2.assert_called_once()


def test_upload_file_calculates_size_correctly():
    """Test that upload_file_to_s3 correctly calculates file size"""
    test_id = uuid4()
    mock_s3 = MagicMock(spec=S3Client)
    
    file_content = b"a" * 1000  # 1000 bytes
    file_obj = io.BytesIO(file_content)
    upload_file = UploadFile(file_obj, filename="test.bin")
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        upload_file_to_s3(upload_file, test_id, mock_s3)
        
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs['ContentLength'] == 1000


def test_download_file_chunks():
    """Test that download handles chunked data correctly"""
    test_id = uuid4()
    chunk1 = b"first chunk "
    chunk2 = b"second chunk"
    
    mock_streaming_body = MagicMock()
    mock_streaming_body.iter_chunks.return_value = [chunk1, chunk2]
    mock_streaming_body.close = MagicMock()
    
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.get_object.return_value = {'Body': mock_streaming_body}
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        result = download_file_from_s3(test_id, mock_s3)
        
        result.seek(0)
        assert result.read() == chunk1 + chunk2


def test_download_file_from_s3_closes_stream_on_success():
    """Test that download_file_from_s3 closes StreamingBody on successful download"""
    test_id = uuid4()
    test_content = b"test content"
    
    mock_streaming_body = MagicMock()
    mock_streaming_body.iter_chunks.return_value = [test_content]
    mock_streaming_body.close = MagicMock()
    
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.get_object.return_value = {'Body': mock_streaming_body}
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        download_file_from_s3(test_id, mock_s3)
        
        mock_streaming_body.close.assert_called_once()


def test_download_file_from_s3_closes_stream_on_error():
    """Test that download_file_from_s3 closes StreamingBody even on error"""
    test_id = uuid4()
    
    mock_streaming_body = MagicMock()
    mock_streaming_body.iter_chunks.side_effect = Exception("Download failed")
    mock_streaming_body.close = MagicMock()
    
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.get_object.return_value = {'Body': mock_streaming_body}
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        with pytest.raises(Exception):
            download_file_from_s3(test_id, mock_s3)
        
        mock_streaming_body.close.assert_called_once()


def test_download_file_from_s3_returns_empty_bytesio_for_disk_write(tmp_path):
    """Test that download_file_from_s3 returns empty BytesIO when writing to disk"""
    test_id = uuid4()
    test_content = b"file content"
    test_file = tmp_path / "test_download.txt"
    
    mock_streaming_body = MagicMock()
    mock_streaming_body.iter_chunks.return_value = [test_content]
    mock_streaming_body.close = MagicMock()
    
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.get_object.return_value = {'Body': mock_streaming_body}
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        result = download_file_from_s3(test_id, mock_s3, filename=str(test_file))
        
        assert isinstance(result, io.BytesIO)
        assert result.read() == b""
        assert test_file.exists()
        assert test_file.read_bytes() == test_content


def test_download_file_from_s3_before_get_object_error():
    """Test that download_file_from_s3 handles error before StreamingBody is assigned"""
    test_id = uuid4()
    mock_s3 = MagicMock(spec=S3Client)
    mock_s3.get_object.side_effect = BotoCoreError()
    
    with patch('app.core.s3.settings') as mock_settings:
        mock_settings.s3_bucket_name = "test-bucket"
        
        with pytest.raises(HTTPException) as exc_info:
            download_file_from_s3(test_id, mock_s3)
        
        assert exc_info.value.status_code == 500
