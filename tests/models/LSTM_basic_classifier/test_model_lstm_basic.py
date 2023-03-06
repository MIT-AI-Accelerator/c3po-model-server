from app.models.LSTM_basic_classifier.model import get_model, Model

#************Mocks*******************
# None
#*************************************

#************Setup*******************
model_1 = Model()
#*************************************

# ensure get_model returns a model
def test_get_model_returns_model():
    assert isinstance(get_model(), Model)

# Given: A line of chat--"Hello there!" and a model
# When: We pass this chat as a string to classify_single_label
# Then: we expect to receive a recycle, action, or review label response
# TODO: figure out how to mock the model using the TF testing framework eventually
def test_classify_single_label_when_string_received_then_label_exists():
    output_label = model_1.classify_single_label("Hello there!")
    assert output_label in ["recycle", "review", "action"]

# Given: A model
# When: We pass None or not str
# Then: we expect an empty string response
# TODO: figure out how to mock the model using the TF testing framework eventually
def test_classify_single_label_when_none_received_then_empty_label_return():
    output_label_1 = model_1.classify_single_label(None)
    output_label_2 = model_1.classify_single_label(1)

    assert output_label_1 == ""
    assert output_label_2 == ""

# Given: A line of chat--"Hello there!" and a model
# When: We pass this chat as a string or as a single-item list to classify_label
# Then: we expect to receive a recycle, action, or review label response as a list of length 1
# TODO: figure out how to mock the model using the TF testing framework eventually
def test_classify_label_when_string_or_single_received_then_label_exists():
    output_labels_1 = model_1.classify_label("Hello there!")
    output_labels_2 = model_1.classify_label(["Hello there!"])

    assert len(output_labels_1) == 1
    assert output_labels_1[0] in ["recycle", "review", "action"]

    assert len(output_labels_2) == 1
    assert output_labels_2[0] in ["recycle", "review", "action"]

# Given: A model
# When: We pass None or not str/list to classify_label
# Then: we expect an empty string response in a list
# TODO: figure out how to mock the model using the TF testing framework eventually
def test_classify_label_when_none_received_then_empty_list_return():
    output_labels_1 = model_1.classify_label(None)
    output_labels_2 = model_1.classify_label(1)

    assert output_labels_1 == [""]
    assert output_labels_2 == [""]

# Given: Two chats and a model
# When: We pass these chat as two-item list to classify_label
# Then: we expect to receive a recycle, action, or review label response as a list of length 2
# TODO: figure out how to mock the model using the TF testing framework eventually
def test_classify_label_when_multi_item_list_received_then_labels_exist():
    output_labels_1 = model_1.classify_label(["Hello there!", "It is raining"])

    assert len(output_labels_1) == 2
    assert output_labels_1[0] in ["recycle", "review", "action"]
    assert output_labels_1[1] in ["recycle", "review", "action"]

# Given: Two chats and a model
# When: We pass these chat as three-item list plus "None" to classify_label
# Then: we expect to receive a recycle, action, or review label response and an empty string as a list of length 3
# TODO: figure out how to mock the model using the TF testing framework eventually
def test_classify_label_when_multi_item_list_received_and_none_then_labels_exist_and_one_empty():
    output_labels_1 = model_1.classify_label(["Hello there!", "It is raining", None])

    assert len(output_labels_1) == 3
    assert output_labels_1[0] in ["recycle", "review", "action"]
    assert output_labels_1[1] in ["recycle", "review", "action"]
    assert output_labels_1[2] == ""
