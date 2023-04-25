
from app.aimodels.bertopic.ai_services.inference.inference_model import load_model
from app.settings.settings import settings
from app.dependencies import httpx_client
import pandas as pd

# see mm api spec here: https://api.mattermost.com/#tag/posts/operation/GetPostsForChannel
# see httpx docs here: https://www.python-httpx.org/
async def get_posts_by_channel(channel_id: str):
    url = f'{settings.mm_base_url}/api/v4/channels/{channel_id}/posts?per_page'

    headers = {'Authorization': f'Bearer {settings.mm_token}'}
    params = {'per_page': '1000'}

    req = httpx_client.build_request('GET', url, headers=headers, params=params)
    return await httpx_client.send(req, stream=True)

def train_bertopic_on_posts(posts: list[dict]):
    # TODO: ensure posts has the correct fields
    df = pd.json_normalize(posts, record_path="posts")

    #TODO: replace with load
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(msgs, show_progress_bar=True)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    topic_model = BERTopic(nr_topics=num_top_trending, vectorizer_model=vectorizer_model).fit(msgs, embeddings)
    topic_model.save(mfile)

    fig1 = topic_model.visualize_documents(msgs, embeddings=embeddings, title='Documents and Topics for ' + cname)
    fig1.write_json(path.join(ppath, 'bert_docs_and_topics_' + cname + '.json'))




    # bert_model = load_model('1')
    # print(request.team)
    # return VisualizationResponse(
    #     plot_params=bert_model.trained_models[0].plotly_bubble_config
    # )


# for later: mapping chats to a specific document
# dinfo = topic_model.get_document_info(msgs)
# docs_per_topics = dinfo.groupby(["Topic"]).apply(lambda x: x.index).to_dict()



    # add in start and end date along with number of topics requested
    # list(channel) and team information
    # list(seed_words)

    # under channel endpoint, chats are stores with message and createat, change to "since"
    # default to user desired params
