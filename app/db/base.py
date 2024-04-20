# Import all the models, so that Base has them before being
# imported by Alembic
# pylint: disable=unused-import
from app.db.base_class import Base  # noqa
from app.aimodels.bertopic.models.bertopic_embedding_pretrained import BertopicEmbeddingPretrainedModel # noqa
from app.aimodels.bertopic.models.bertopic_trained import BertopicTrainedModel # noqa
from app.aimodels.bertopic.models.document import DocumentModel # noqa
from app.aimodels.bertopic.models.document_embedding_computation import DocumentEmbeddingComputationModel # noqa
from app.aimodels.bertopic.models.document_bertopic_trained_model import DocumentBertopicTrainedModel # noqa
from app.aimodels.bertopic.models.topic import TopicSummaryModel # noqa
from app.aimodels.bertopic.models.bertopic_visualization import BertopicVisualizationModel # noqa
from app.aimodels.gpt4all.models.llm_pretrained import LlmPretrainedModel # noqa
from app.mattermost.models.mattermost_channels import MattermostChannelModel # noqa
from app.mattermost.models.mattermost_users import MattermostUserModel # noqa
from app.mattermost.models.mattermost_documents import MattermostDocumentModel # noqa
