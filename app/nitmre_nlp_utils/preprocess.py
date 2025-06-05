import re
import pandas as pd
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
    p = re.compile('[^\w/#@]+')
    return set(p.split(msg))  # only keep unique tokens to avoid revisiting during preprocessing


def _acronym_repl_helper(m: re.Match, token_expanded: str) -> str:
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
    msg_only: bool=False,
) -> str | tuple[str, list[str], list[str]]:
    """Extract all tokens from a message and replace instances with expanded
    acronyms.

    Also extracts RCH call signs and ICAO codes."""
    tokens = _tokenize(msg)

    # Any acronyms found that match a token will be expanded;
    # matches are case-sensitive.
    msg_expanded = msg
    call_signs, icaos = [], []
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
                lambda m: _acronym_repl_helper(m, token_expanded),
                msg_expanded)

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


def _thread_message(
    root_message: str,
    df_thread: pd.DataFrame,
    message_col_name: str,
) -> str:
    """Take a root message and a dataframe of messages in the root's thread to
    create a single threaded message.

    Use message_col_name to specify the column that contains the messages in
    the thread."""
    messages = [root_message]

    for _, row in df_thread.iterrows():
        message = row[message_col_name]
        messages.append(message)

    return '\n'.join(messages)


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
    
    # Separate the root message from messages in the thread.
    df_roots, df_threads = df[df['root_id'] == ''], df[df['root_id'] != '']

    for root_id in tqdm(df_roots['id'].to_list()):
        df_root = df_roots[df_roots['id'] == root_id]
        df_thread = df_threads[df_threads['root_id'] == root_id].copy()
        
        # Preserve chronological ordering of each message in the thread.
        df_thread.sort_values(by='create_at', ascending=True, inplace=True)

        threaded_message = _thread_message(
            df_root.iloc[0][message_col_name], df_thread, message_col_name)
        
        # Replace the original message with the thread.
        df_roots.loc[
            df_roots['id'] == root_id, message_col_name
        ] = threaded_message
    
    return df_roots
