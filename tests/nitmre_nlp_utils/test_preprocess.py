import pytest
import random
import pandas as pd
import numpy as np

from datetime import datetime
from pandas.testing import assert_frame_equal

from app.nitmre_nlp_utils import preprocess as pre


def generate_random_timestamp(
    start_dt: datetime = datetime(2000, 1, 1),
    end_dt: datetime = datetime(2024, 12, 31),
) -> datetime:
    sts = start_dt.timestamp()
    ets = end_dt.timestamp()

    random_dt = random.uniform(sts, ets)
    return datetime.fromtimestamp(random_dt)


def test_threading_successful():
    df = pd.DataFrame(
        {
            'id': [0, 1, 2, 3, 4, 5, 7, 9, 10, 11],
            'root_id': ['', np.nan, 0, 0, 1, '', 6, 8, 8, 8],
            'create_at': [
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 4, 1),
                datetime(2024, 3, 1),
                datetime(2024, 5, 1),
                datetime(2024, 6, 1),
                datetime(2024, 8, 1),
                datetime(2024, 10, 1),
                datetime(2024, 9, 1),
                datetime(2024, 11, 1),
            ],
            'message': [
                'zero',
                'one',
                'two',
                'three',
                'four',
                'five',
                'seven',
                'nine',
                'ten',
                'eleven',
            ],
        }
    )

    expected_root_ids = (0, 1, 5, 7, 10)
    expected = pd.DataFrame(
        {
            'id': expected_root_ids,  # empty string and nans collapse to root id
            'root_id': ['', np.nan, '', '', ''],
            'create_at': df[df['id'].isin(expected_root_ids)]['create_at'],  # type: ignore
            'message': [
                'zero\nthree\ntwo',  # threads are sorted by create_at
                'one\nfour',  # threaded messages are delimited by newlines
                'five',  # keep messages that don't have a thread
                'seven',  # Threads with a missing root message aren't lost...
                'ten\nnine\neleven',  # ...even with more than one message in the thread.
            ],
        }
    )
    result = pre.convert_conversation_threads(df, 'message')
    assert_frame_equal(result, expected)

    result_processes_2 = pre.convert_conversation_threads(df, 'message', 2)
    assert_frame_equal(result_processes_2, expected)

    result_processes_3 = pre.convert_conversation_threads(df, 'message', 3)
    assert_frame_equal(result_processes_3, expected)


def test_threading_incorrect_columns():
    message_col_name = 'messages'
    num_rows = 5

    df = pd.DataFrame(
        {
            'idx': np.arange(num_rows),
            'root-id': np.arange(num_rows - 1, -1, -1),
            'created_at': list(
                generate_random_timestamp() for _ in range(num_rows)
            ),
            message_col_name: list(str(i) for i in range(num_rows)),
        }
    )

    with pytest.raises(KeyError):
        pre.convert_conversation_threads(df, message_col_name)


def test_threading_empty_dataframe():
    df = pd.DataFrame(columns=['id', 'root_id', 'create_at', 'message'])

    expected = pd.DataFrame(columns=['id', 'root_id', 'create_at', 'message'])
    result = pre.convert_conversation_threads(df, 'message')
    assert_frame_equal(result, expected)


def test_threading_duplicate_messages():
    df = pd.DataFrame(
        {
            'id': [0, 1, 2, 3, 4, 5],
            'root_id': ['', '', 0, 0, 1, 1],
            'create_at': [
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 3, 1),
                datetime(2024, 3, 1),  # Duplicate timestamp
                datetime(2024, 4, 1),
                datetime(2024, 4, 1),  # Duplicate timestamp
            ],
            'message': [
                'zero',
                'one',
                'two',
                'two',  # Duplicate message
                'four',
                'four',  # Duplicate message
            ],
        }
    )

    expected_root_ids = (0, 1)
    expected = pd.DataFrame(
        {
            'id': expected_root_ids,
            'ro'
            'ot_id': ['', ''],
            'create_at': df[df['id'].isin(expected_root_ids)]['create_at'],  # type: ignore
            'message': [
                'zero\ntwo\ntwo',  # Duplicate messages are preserved
                'one\nfour\nfour',  # Duplicate messages are preserved
            ],
        }
    )
    result = pre.convert_conversation_threads(df, 'message')
    assert_frame_equal(result, expected)


def test_preprocess_message():
    acronym_dictionary = {
        "LL": "Logistics Liaison"
    }
    icao_dictionary = {
        "KATL": "Atlanta International Airport",
        "KLAX": "Los Angeles International Airport",
    }

    # Test input message
    msg = (
        "Message from @DO1 regarding RCH123. "
        "Flight is scheduled to depart KLAX. "
        "LL coordination required. "
    )

    # Expected outputs
    expected_msg_only = (
        "Message from @DO1 regarding RCH123. "
        "Flight is scheduled to depart KLAX. "
        "Logistics Liaison coordination required. "
    )
    expected_full_output = (
        expected_msg_only,
        ["RCH123"],  # Extracted RCH call signs
        ["KLAX"],  # Extracted ICAO codes
    )

    # Test msg_only=True
    result_msg_only = pre.preprocess_message(acronym_dictionary, icao_dictionary, msg, msg_only=True)
    assert result_msg_only == expected_msg_only, f"Expected: {expected_msg_only}, Got: {result_msg_only}"

    # Test full output (msg_only=False)
    result_full_output = pre.preprocess_message(acronym_dictionary, icao_dictionary, msg, msg_only=False)
    assert result_full_output == expected_full_output, f"Expected: {expected_full_output}, Got: {result_full_output}"
