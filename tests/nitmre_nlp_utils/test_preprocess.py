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
