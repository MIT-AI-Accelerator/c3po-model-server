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
**NOTE**: Temporary issue between arm64 and x86...for now, x86 should install via `pip install -r requirements.txt`.

8.  In `c3po-model-server/app/core/env_var`, create a `secrets.env` file and ensure it is on the `.gitignore`.  Add the following for local dev:
```sh
MINIO_BUCKET_NAME=test
MINIO_ACCESS_KEY=miniouser
MINIO_SECRET_KEY=minioadmin
MINIO_ENDPOINT_URL=localhost:9000
MINIO_SECURE=False

POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_DB=postgres

MM_TOKEN="<your_preprod_mattermost_token>"
```

9. Launch postgres, pgadmin, and minio via docker-compose `docker-compose up --build`.

10. Visit `localhost:9001`.  Login with user:`miniouser` and password:`minioadmin`.  This is the minio console.

11. Visit `localhost:5050`.  Login with email:`user@test.com` and password:`admin`.  This is the pgadmin console.  **See notes below for important details**

12. Run the app db init script `./scripts/prestart.sh`

13. Keeping your docker containers running, start the app in a new terminal (activate your conda env first) with `pipenv run uvicorn app.main:versioned_app --reload`.

14. Open `localhost:8000/v1/docs` and start interacting with swagger!

15. Run tests and get coverage with `pytest --cov=app`, and get html reports for vs code live server (or any server) with `pytest --cov --cov-report=html:coverage_re`

16.  You can shut down and your db / minio data will persist via docker volumes.


# Notes
- This codebase assumes that you start from a base tensorflow Docker image or are running tensorflow locally via conda.  We do not install tensorflow via pip.  All other dependencies are install via pip.
- You will see that `POSTGRES_SERVER=localhost` in the above steps, however, make sure that you login with hostname `db` in pgAdmin (under the "connection" tab in server properties).  This is because the pgAdmin container is launched in the same docker network as the postgres container, so it uses the service name, whereas launching this app from command line uses port forwarding to localhost.  The user, password, and db name will all be `postgres`, port `5432`.
- For basic CRUD, you can follow this format:
```
from .base import CRUDBase
from app.models.item import Item
from app.schemas.item import ItemCreate, ItemUpdate

item = CRUDBase[Item, ItemCreate, ItemUpdate](Item)
```

# Knowledge and helpful links
## Tools for this repo
- [Tutorial followed-ish for this repo](https://curiousily.com/posts/deploy-bert-for-sentiment-analysis-as-rest-api-using-pytorch-transformers-by-hugging-face-and-fastapi/)
- [Install conda and tensorflow on Mac M1](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706)
- [`pipenv` with `conda`](https://stackoverflow.com/questions/50546339/pipenv-with-conda)
- [Basics of `pipenv` for application dependency management](https://python.plainenglish.io/getting-started-with-pipenv-d224328799de)
- [Conda and pipenv cheat sheet](https://gist.github.com/ziritrion/8024025672ea92b8bdeb320d6015aa0d)
- [How to use pre-commit framework for git hooks](https://pre-commit.com/index.html)

## Testing
In general, tensorflow and pytorch use the underlying `unittest` framework that comes stock with Python.  However, FastAPI has a ton of great features through `pytest` that make testing HTTP much, much easier.  Good news is that, for the most part, pytest as the runner will also handle unittest, so we can use the TF or pytorch frameworks with unittest and FastAPI with pytest.  Some articles on this:
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Tensorflow testing](https://theaisummer.com/unit-test-deep-learning/)
- [Pytest handling unittest](https://docs.pytest.org/en/latest/how-to/unittest.html#pytest-features-in-unittest-testcase-subclasses)
- [Mocking in pytest--especially import location](https://changhsinlee.com/pytest-mock/)
- [Test coverage using `coverage`](https://coverage.readthedocs.io/en/7.2.1/)


## Tools for git
- [Storing Credentials](https://git-scm.com/docs/git-credential-store)...or just type `git config --global credential.helper store`
- [GPG Commit Signing](https://confluence.il2.dso.mil/display/afrsba/Setting+up+GPG+for+GitLab+Commit+Signing)

## Tools for Docker
- [Deleting Volumes](https://forums.docker.com/t/where-are-volumes-located-on-os-x/10488)
- [Setting up pgAdmin in Docker](https://belowthemalt.com/2021/06/09/run-postgresql-and-pgadmin-in-docker-for-local-development-using-docker-compose/)
- [Setting up postgreSQL for FastAPI in docker](https://github.com/tiangolo/full-stack-fastapi-postgresql/blob/master/%7B%7Bcookiecutter.project_slug%7D%7D/docker-compose.yml)
- [Full FastAPI / postgres / docker tutorial](https://www.jeffastor.com/blog/pairing-a-postgresql-db-with-your-dockerized-fastapi-app)

# P1 Links
## Basic Links
- [P1 Code repo](https://code.il4.dso.mil/platform-one/products/ai-accel/transformers/c3po-model-server)
- [P1 Pipelines](https://code.il4.dso.mil/platform-one/products/ai-accel/transformers/c3po-model-server/-/pipelines)
- [Padawan Docs (landing pages)](https://padawan-docs.dso.mil/)

## DevOps Links
### SonarQube
- [SonarQube for dependency check](https://sonarqube.il4.dso.mil/dashboard?id=platform-one-products-ai-accel-transformers-c3po-model-server-dependencies)
- [SonarQube for SCA & code coverage](https://sonarqube.il4.dso.mil/dashboard?id=platform-one-products-ai-accel-transformers-c3po-model-server)
- [False Positive Clearing SQ and trufflehog](https://confluence.il2.dso.mil/display/PUCKBOARD/Sonarqube+False-Positive+Issue+Workflow)
- [Argo page](https://argocd-il4.admin.dso.mil/applications/argocd/p1-il4-mission-staging-transformers-transformers?view=tree&resource=)

### K8s configs
- [IL4 mission bootstrap](https://code.il4.dso.mil/platform-one/devops/mission-bootstrap/il4-mission-bootstrap/-/tree/master/integrations/ai-accel/transformers)

## Twistlock
- [Twistlock link for repo](https://twistlock-il4.admin.dso.mil/api/v1/platform-one/products/ai-accel/transformers/c3po-model-server)
- [Twistlock errors](https://confluence.il2.dso.mil/display/P1MDOHD/TS+-+Twistlock+-+Stage+Failure#TSTwistlockStageFailure-400:NoRegistrySettingsSpecificationApply)

## Iron Bank
- [Baseline image for this project](https://ironbank.dso.mil/repomap/details;registry1Path=opensource%252Ftensorflow%252Ftensorflow-2.5.1)
- [Code for baseline image](https://repo1.dso.mil/dsop/opensource/tensorflow/tensorflow-2.5.1/-/blob/development/Dockerfile)

## Helpdesk Links
- [Jira Service Desk](https://jira.il2.dso.mil/servicedesk/customer/portals)
- [Add someone to Mattermost, Confluence, Jira](https://jira.il2.dso.mil/servicedesk/customer/portal/1/create/498?q=access&q_time=1673363010205)
- [Add someone to DevOps](https://jira.il2.dso.mil/servicedesk/customer/portal/73/create/706?q=access&q_time=1673363566291)
- [Request a pipeline](https://jira.il2.dso.mil/servicedesk/customer/portal/73/group/240)
- [COT Ticket for AIA](https://jira.il2.dso.mil/browse/COT-484)
- [False positive clearing](https://jira.il2.dso.mil/servicedesk/customer/portal/73/create/730)
