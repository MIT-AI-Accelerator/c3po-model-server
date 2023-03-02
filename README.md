# Getting Started Locally
1. Make sure that you have `conda` installed.  [Recommend this article](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706) if you are using an M1-based Mac for development.

2. Create and activate a new conda environment `c3po-os-api` with python 3.9
```bash
conda create --name c3po-os-api python=3.9
conda activate c3po-os-api
```

3. Install tensorflow 2.X into your `conda` environment.  Again, [follow the steps at this article](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706) if you are using an M1-based Mac.

4. Run `which pip` and `which python` to verify path to make sure that your `python` and `pip` binaries are coming from your `conda` virtual environment.  Note that the order in which you install conda vs. pip matters to set virtual env priorities.

5. Install `pipenv`: `pip install pipenv`

6. Setup `pipenv` to shadow `conda`-installed packages and local env version of python: `pipenv --python=$(which python) --site-packages`

7. Install dependencies and dev dependencies locally from the `Pipfile` by running `pipenv install --dev`.

8. Start the app `pipenv run uvicorn app.api:app`.

9. Post a request in a separate shell!
```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"text": "A fun little note."}' \
    http://localhost:8000/predict
```

# Notes
- This codebase assumes that you start from a base tensorflow Docker image or are running tensorflow locally via conda.  We do not install tensorflow via pip.  All other dependencies are install via pip.

# Knowledge and helpful links
- [Tutorial followed-ish for this repo](https://curiousily.com/posts/deploy-bert-for-sentiment-analysis-as-rest-api-using-pytorch-transformers-by-hugging-face-and-fastapi/)
- [Install conda and tensorflow on Mac M1](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706)
- [`pipenv` with `conda`](https://stackoverflow.com/questions/50546339/pipenv-with-conda)
- [Basics of `pipenv` for application dependency management](https://python.plainenglish.io/getting-started-with-pipenv-d224328799de)
- [Conda and pipenv cheat sheet](https://gist.github.com/ziritrion/8024025672ea92b8bdeb320d6015aa0d)
- [How to use pre-commit framework for git hooks](https://pre-commit.com/index.html)

# P1 Links
- [P1 Code repo](https://code.il4.dso.mil/platform-one/products/ai-accel/transformers/c3po-model-server)
- [P1 Pipelines](https://code.il4.dso.mil/platform-one/products/ai-accel/transformers/c3po-model-server/-/pipelines)
- [Jira Service Desk](https://jira.il2.dso.mil/servicedesk/customer/portals)
- [Add someone to Mattermost, Confluence, Jira](https://jira.il2.dso.mil/servicedesk/customer/portal/1/create/498?q=access&q_time=1673363010205)
- [Add someone to DevOps](https://jira.il2.dso.mil/servicedesk/customer/portal/73/create/706?q=access&q_time=1673363566291)
- [COT Ticket for AIA](https://jira.il2.dso.mil/browse/COT-484)
