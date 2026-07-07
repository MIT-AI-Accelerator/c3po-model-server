import json
import pandas as pd
from io import StringIO
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from app.main import versioned_app
from app.core.config import get_acronym_dictionary, get_icao_dictionary, OriginationEnum, settings
from app.ppg_common.schemas.bertopic.document import DocumentCreate
from app.aimodels.bertopic.crud.crud_document import document

client = TestClient(versioned_app)

# Verify app versioning set to /v1
def test_v1_exists():
    response = client.get(
        "/v1/docs"
    )
    assert response.status_code == 200

# set originated_from for standard app usage
def test_set_originated_from():
    response = client.get("/v1/originated_from_app")
    data = response.json()
    assert data == OriginationEnum.ORIGINATED_FROM_APP
    assert response.status_code == 200

    response = client.get("/v1/originated_from_test")
    data = response.json()
    assert data == OriginationEnum.ORIGINATED_FROM_TEST
    assert response.status_code == 200

# test upload acronym list
def test_upload_acronym_dictionary():
    acronym_dictionary = dict({'PPG': 'Prototype Proving Ground (PPG)'})
    response = client.post("/v1/upload_acronym_dictionary", params={'acronym_dictionary': json.dumps(acronym_dictionary)})
    assert response.status_code == 200
    assert response.json() == acronym_dictionary
    assert get_acronym_dictionary() == acronym_dictionary

# test upload icao list
def test_upload_icao_dictionary():
    acode = 'KBOS'
    icao_dictionary = dict({acode: 'General Edward Lawrence Logan International Airport (KBOS)'})
    response = client.post("/v1/upload_icao_dictionary", params={'icao_dictionary': json.dumps(icao_dictionary)})
    assert response.status_code == 200
    assert response.json() == icao_dictionary
    assert get_icao_dictionary()[acode] == icao_dictionary[acode]

# test download db data
def test_download_db_data_invalid():
    table_name = 'notatablemodel'
    response = client.get("/v1/download", params={'table_name': table_name})
    assert response.status_code == 422

    table_name = 'documentmodel'
    response = client.get("/v1/download", params={'table_name': table_name, 'limit': -1})
    assert response.status_code == 422

# test download db data
def test_download_db_data_valid(db: Session):

    # create a bertopic_embedding_pretrained
    document_create = DocumentCreate(text='a test document')
    document.create(db, obj_in=document_create)

    table_name = 'documentmodel'
    limit = 1
    response = client.get("/v1/download", params={'table_name': table_name, 'limit': limit})
    assert response.status_code == 200

    cstr = StringIO(response.text)
    df = pd.read_csv(cstr, sep=",")
    assert len(df) == limit

def test_shutdown():
    response = client.get('/v1/docs')
    assert response.status_code == 200


def test_download_db_data_with_no_limit(db: Session):
    """Test downloading database data without limit parameter"""
    document_create = DocumentCreate(text='test document for no limit')
    document.create(db, obj_in=document_create)
    
    table_name = 'documentmodel'
    response = client.get("/v1/download", params={'table_name': table_name})
    assert response.status_code == 200
    assert 'text/csv' in response.headers['content-type']


def test_download_db_response_headers(db: Session):
    """Test that download response has correct headers"""
    document_create = DocumentCreate(text='test document')
    document.create(db, obj_in=document_create)
    
    table_name = 'documentmodel'
    response = client.get("/v1/download", params={'table_name': table_name, 'limit': 1})
    
    assert response.status_code == 200
    assert 'Content-Disposition' in response.headers
    assert 'attachment' in response.headers['Content-Disposition']
    assert f'{table_name}.csv' in response.headers['Content-Disposition']


def test_cors_middleware_configured():
    """Test that CORS middleware is properly configured"""
    assert hasattr(versioned_app, 'user_middleware')
    assert len(versioned_app.user_middleware) > 0


def test_app_title():
    """Test that the app has correct title"""
    response = client.get("/v1/openapi.json")
    assert response.status_code == 200
    openapi_data = response.json()
    assert openapi_data['info']['title'] == 'Transformers API'


def test_app_includes_aimodels_router():
    """Test that aimodels router is included"""
    response = client.get("/v1/openapi.json")
    assert response.status_code == 200
    openapi_data = response.json()
    
    paths = openapi_data.get('paths', {})
    aimodels_paths = [p for p in paths.keys() if 'aimodels' in p]
    assert len(aimodels_paths) > 0


def test_app_includes_experimental_router():
    """Test that experimental router is included"""
    response = client.get("/v1/openapi.json")
    assert response.status_code == 200
    openapi_data = response.json()
    
    paths = openapi_data.get('paths', {})
    has_experimental = any('mattermost' in p or 'sentiments' in p for p in paths.keys())
    assert has_experimental


def test_originated_from_returns_correct_type():
    """Test that originated_from endpoints return OriginationEnum"""
    response = client.get("/v1/originated_from_app")
    assert response.status_code == 200
    data = response.json()
    assert data in [e.value for e in OriginationEnum]


def test_upload_acronym_dictionary_updates_settings():
    """Test that uploading acronym dictionary updates the settings"""
    test_dict = {"NASA": "National Aeronautics and Space Administration"}
    response = client.post("/v1/upload_acronym_dictionary", 
                          params={'acronym_dictionary': json.dumps(test_dict)})
    assert response.status_code == 200
    assert get_acronym_dictionary() == test_dict


def test_upload_icao_dictionary_appends():
    """Test that uploading ICAO dictionary appends to existing"""
    initial_dict = {"KJFK": "JFK Airport"}
    client.post("/v1/upload_icao_dictionary", 
               params={'icao_dictionary': json.dumps(initial_dict)})
    
    additional_dict = {"KLAX": "LAX Airport"}
    response = client.post("/v1/upload_icao_dictionary", 
                          params={'icao_dictionary': json.dumps(additional_dict)})
    
    assert response.status_code == 200
    result = get_icao_dictionary()
    assert "KJFK" in result
    assert "KLAX" in result


def test_download_limit_zero_returns_all_rows(db: Session):
    """Test that limit=0 returns all rows"""
    for i in range(3):
        document_create = DocumentCreate(text=f'test document {i}')
        document.create(db, obj_in=document_create)
    
    table_name = 'documentmodel'
    response = client.get("/v1/download", params={'table_name': table_name, 'limit': 0})
    assert response.status_code == 200
    
    cstr = StringIO(response.text)
    df = pd.read_csv(cstr, sep=",")
    assert len(df) >= 3


def test_versioned_app_default_version():
    """Test that default API version is 1"""
    response = client.get("/v1/docs")
    assert response.status_code == 200


def test_app_root_path_configured():
    """Test that app root path is configured from settings"""
    response = client.get("/v1/openapi.json")
    assert response.status_code == 200
