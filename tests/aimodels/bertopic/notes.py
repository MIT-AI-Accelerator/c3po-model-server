# [{"text":"document 1"},{"text":"document 2"},{"text":"document 3"},{"text":"document 4"},{"text":"document 5"},{"text":"document 6"},{"text":"document 7"},{"text":"document 8"},{"text":"document 9"},{"text":"document 10"}]
# {
#   "bertopic_embedding_pretrained_id": "85d33b8f-a436-4a20-aed4-243e01db8660",
#   "document_ids": ["3bda7e49-cb03-4549-a37c-1b84353a3740","c70cd389-1f54-457c-ac30-593a88ed5e5f","1424ca7d-be01-4a27-85b1-d8aa79fc321f","bfbeb67c-4137-4110-8f27-fa8168b6853f","6cb5a1bf-3e2e-44d7-a611-12c472f16c06","2b121c6a-f198-444a-a1d2-97e929e1896c","cc0a8ea8-44b5-484d-8bee-1ec1660a1560","860081cf-509d-4d24-b982-8c0758634aa3","73473b4e-1245-4c32-982f-7314e928add8","5f17a8e4-ec9b-41cd-b3c7-be1cc8a008e7"]
# }
    # new_plotly_bubble_config = {
    #     "data": [
    #         {
    #             "type": "scatter",
    #             "x": [1, 2, 3],
    #             "y": [3, 1, 6]
    #         }
    #     ],
    #     "layout": {"title": {"text": "A Fancy Plot"}}
    # }

    # save calculated embeddings computations
    # if inference_output.updated_document_indicies:
    #     for i, wasUpdated in enumerate(inference_output.updated_document_indicies):
    #         if (wasUpdated):
    #             document_id = documents[i].id
    #             bertopic_embedding_pretrained_id = request.bertopic_embedding_pretrained_id
    #             embedding_vector = inference_output.embeddings[i]
    #             crud.document_embedding_computation.create(db, obj_in=DocumentEmbeddingComputationCreate(
    #                 document_id=document_id,
    #                 bertopic_embedding_pretrained_id=bertopic_embedding_pretrained_id,
    #                 embedding_vector=embedding_vector
    #             ))
