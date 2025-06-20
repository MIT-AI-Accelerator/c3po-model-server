import json
import pandas as pd
from io import StringIO
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from app.main import versioned_app
from app.core.config import get_acronym_dictionary, get_icao_dictionary, OriginationEnum
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
