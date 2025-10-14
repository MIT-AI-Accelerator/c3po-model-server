import re
import pandas as pd
import numpy as np
from tqdm import tqdm


def _tokenize(msg: str) -> set[str]:
    """Custom tokenizer based on observed message patterns.

    In addition to alphanumeric characters, we choose to preserve these
    characters when splitting tokens:
    @   used to tag users (e.g., @DO1)
    #   used to tag the message (e.g., #RCH123)
    /   often used between acronyms in an explicit/implied coordination
        line (e.g., DO/LL)
    """
    p = re.compile(r'[^\w/#@]+')
    return set(
        p.split(msg)
    )  # only keep unique tokens to avoid revisiting during preprocessing


def _acronym_repl_helper(m: re.Match[str], token_expanded: str) -> str:
    """Helper function that allows us to preserve plurality when expanding
    acronyms.

    Can be extended with compiled regex to do additional processing on
    acronyms."""
    result = f' {token_expanded}'
    return result + 's' if m.group(0)[-1] == 's' else result


def preprocess_message(
    acronym_dictionary: dict[str, str],
    icao_dictionary: dict[str, str],
    msg: str,
    *,
    msg_only: bool = False,
) -> str | tuple[str, list[str], list[str]]:
    """Extract all tokens from a message and replace instances with expanded
    acronyms.

    Also extracts RCH call signs and ICAO codes."""
    tokens = _tokenize(msg)

    # Any acronyms found that match a token will be expanded;
    # matches are case-sensitive.
    msg_expanded = msg
    call_signs: list[str] = []
    icaos: list[str] = []
    for token in tokens:
        # compile the token as a regex pattern
        p = re.compile(r'{}'.format(token))
        s = p.search(msg_expanded)
        if s is None:
            continue

        # ICAO codes
        elif token in icao_dictionary:
            icaos.append(token)

        # Acronyms
        elif token in acronym_dictionary:
            # Only match acronyms that are capitalized and have leading
            # whitespace. Will also match a trailing lowercase 's'.
            token_expanded = acronym_dictionary[token]
            p = re.compile(r'\s' + r'{}'.format(token) + r's?')
            msg_expanded = p.sub(
                lambda m: _acronym_repl_helper(m, token_expanded), msg_expanded
            )

        # RCH call signs
        else:
            p = re.match(r'#?RCH\d+', token, re.IGNORECASE)
            if p is None:
                continue

            # Will lose some matches (e.g., RCH123/456/789 only keeps RCH123).
            call_sign = p.group(0)

            # To ensure consistent formatting.
            call_sign = call_sign.lstrip('#').upper()

            # Limit to the longest observed call sign for performance; this
            # length is set in schema field configurations in Milvus.
            call_signs.append(call_sign[:8])

    return msg_expanded if msg_only else (msg_expanded, call_signs, icaos)


def _sep_roots_and_threads(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_roots = df[(df['root_id'] == '') | (df['root_id'].isna())].copy(
        deep=True
    )
    df_threads = df[(df['root_id'] != '') & ~(df['root_id'].isna())].copy(
        deep=True
    )

    # To make sure the thread is in chronological order.
    df_threads.sort_values(by='create_at', ascending=True, inplace=True)  # type: ignore

    return df_roots, df_threads


def _fix_missing_roots(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the dataframe so that missing root_id messages don't cause their
    threads to be discarded.

    We might lose the root message for various reasons (e.g., weak learner,
    root message dated outside of dataset's date range), but we don't want to
    lose those messages.

    This fix takes the earliest message in each thread (by create_at value), and
    sets that id as the new root_id for all messages in the thread.
    """
    df_roots, df_threads = _sep_roots_and_threads(df)

    # Identify the ids of the root messages that are not in the dataframe
    root_ids = set(df_roots['id'])
    thread_root_ids = set(df_threads['root_id'])
    missing_root_ids = thread_root_ids - root_ids
    df_missing = df_threads[df_threads['root_id'].isin(missing_root_ids)]  # type: ignore
    print(df_missing)

    # Take the earliest message in each thread and make that message the root
    idx_min = df_missing.groupby('root_id')['create_at'].transform('idxmin')  # type: ignore
    new_root_ids = df_missing.loc[idx_min, 'id'].values  # type: ignore
    df_missing.loc[:, 'root_id'] = new_root_ids
    df_missing.loc[:, 'root_id'] = np.where(
        df_missing['id'] == df_missing['root_id'], '', df_missing['root_id']
    )

    # Update the original dataframe
    df.update(df_missing)  # type: ignore

    return df


def convert_conversation_threads(
    df: pd.DataFrame,
    message_col_name: str,
) -> pd.DataFrame:
    """Take a full set of individual messages to form single message threads.

    The returned dataframe contains only root message ids, with it's given
    message column containing the full message thread.

    The input dataframe must contain the following columns:
    - id
    - root_id
    - create_at
    - the given message_col_name
    """
    req_cols_set = {'id', 'root_id', 'create_at', message_col_name}
    if len(req_cols_set & set(df.columns)) != len(req_cols_set):
        raise KeyError(f'Invalid dataframe. Required columns: {req_cols_set}.')

    print('Converting raw messages to conversation threads.')

    df = _fix_missing_roots(df)
    df_roots, df_threads = _sep_roots_and_threads(df)

    # Grouping and mapping messages outside of the loop is more efficient than
    # filtering for each root id in every iteration.
    grouped_threads = df_threads.groupby('root_id')[message_col_name].apply(  # type: ignore
        list
    )
    root_messages: dict[str, str] = df_roots.set_index('id')[  # type: ignore
        message_col_name
    ].to_dict()

    for root_id in tqdm(df_roots['id'].to_list()):
        # Messages without a thread are untouched.
        if root_id in grouped_threads:
            root_message = root_messages[root_id]
            thread_messages = grouped_threads[root_id]
            all_messages = [root_message] + thread_messages
            threaded = '\n'.join(all_messages)

            # Replace the original message with the thread.
            df_roots.loc[df_roots['id'] == root_id, message_col_name] = threaded

    return df_roots  # rows are in their original order
