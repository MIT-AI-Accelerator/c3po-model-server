import os
import json
from pydantic import BaseModel

class BertopicEmbeddingModel(BaseModel):
    id: str
    version: int
    sha: str
    uploaded: bool = False

class BertopicTrainedModel(BaseModel):
    id: str
    version: int = 0
    visualization_config: dict = {}

class BertopicModel(BaseModel):
    id: str = '0'
    latest_version: int = 0
    embedding_models: list[BertopicEmbeddingModel] = []
    trained_models: list[BertopicTrainedModel] = []

def load_model(id: str) -> BertopicModel:
    output = BertopicModel()
    output.trained_models.append(BertopicTrainedModel(id=id, version=1, visualization_config=load_vis_config()))

    return output

def load_vis_config() -> dict:
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/vis1.json'), 'rt', encoding='utf-8') as f:
        data = json.load(f)
        return data
