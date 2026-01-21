# Setup and run c3po-model-server
## Start here if you want to use conda (not required)

1. Make sure that you have `conda` installed.  [Recommend this article if on Mac, just do through step 2](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706).

2. Create and activate a new conda environment, e.g., `transformers-api` with python 3.12.
```bash
conda create --name transformers-api python=3.12
conda activate transformers-api
```

3. Run `which pip` and `which python` to verify path to make sure that your `python` and `pip` binaries are coming from your `conda` virtual environment.  Note that the order in which you install conda vs. pip matters to set virtual env priorities.

## Start here if not using conda, just make sure you have the right version of python and pip installed
This procedure installs the necessary packages, builds and runs the containers.

1. Install `poetry` version 1.8.5: `pip install poetry==1.8.5` (or use `pipx` [on link here](https://python-poetry.org/docs/1.4#installing-with-pipx) if you prefer isolated envs and you don't have `conda` managing your env)

2. Create and enter the virtual environment: `poetry shell`. Note: if you use conda, this step may not be necessary.

3. Install the dependencies `poetry install`.

4. Launch postgres, pgadmin, and minio via docker-compose `docker compose up --build`. Note: `docker-compose up --build` may also work, but Docker Compose V1 is deprecated.

## Getting Started Locally
This procedure runs the c3po-model-server without connecting to Mattermost.

1. Run the app db init script `ENVIRONMENT=integration ./scripts/init.sh`

2. Keeping your docker containers running, start the app in a new terminal (activate your conda env first) with `ENVIRONMENT=integration uvicorn app.main:versioned_app --reload`.

3. Run tests and get coverage with `ENVIRONMENT=integration pytest --cov`, and get html reports for vs code live server (or any server) with `ENVIRONMENT=integration pytest --cov --cov-report=html:coverage_re`

4.  You can shut down and your db / minio data will persist via docker volumes.

## Getting Started with Mattermost
This procedure runs the c3po-model-server with a Mattermost connection.
Proceed only if the nitmre-bot token has been shared with you directly.

1.  In `c3po-model-server/app/core/env_var`, create a `secrets.env` file and ensure it is on the `.gitignore`.  Add the following for local dev:
```sh
MM_TOKEN="<your__mattermost_token>"
```
2. Run the app db init script `ENVIRONMENT=local ./scripts/init.sh`

3. Keeping your docker containers running, start the app in a new terminal (activate your conda env first) with `ENVIRONMENT=local uvicorn app.main:versioned_app --reload`.

4. Run tests and get coverage with `ENVIRONMENT=local pytest --cov`, and get html reports for vs code live server (or any server) with `ENVIRONMENT=local pytest --cov --cov-report=html:coverage_re`

# Tools
## Swagger UI

1. Confirm the app is running. Note: the default port is 8000; if you launched the app on an alternative port update the URL in step 2 accordingly.

2. Open `localhost:8000/v1/docs` and start interacting with swagger! Note: if the app is running on dgwonub22, you'll nagivate to this link instead `http://172.25.252.52:8000/v1/docs`.

## Minio
1. Confirm the docker containers are running.

2. Visit `localhost:9001`.  Login with user:`miniouser` and password:`minioadmin`.  This is the minio console.

## PgAdmin

1. Confirm the docker containers are running.

2. Visit `localhost:5050`.  Login with email:`user@test.com` and password:`admin`.  This is the pgadmin console.  **See notes below for important details**

- You will see that `POSTGRES_SERVER=localhost` in the above steps, however, make sure that you login with hostname `db` in pgAdmin (under the "connection" tab in server properties).  This is because the pgAdmin container is launched in the same docker network as the postgres container, so it uses the service name, whereas launching this app from command line uses port forwarding to localhost.  The user, password, and db name will all be `postgres`, port `5432`.

## Poetry
### Adding a package
Note: instructions included in [tutorial linked here](https://realpython.com/dependency-management-python-poetry/)
1. Add the package, e.g., `poetry add transformers` or `poetry add transformers --group <group_name>` where `<group_name>` is the dependency group name, e.g., `test` or `dev`.
2. Update the lockfile with `poetry lock` or `poetry lock --no-update` if you don't want poetry to try to update other deps within your existing versioning constraints
3. Install the packages with `poetry install`, exclude certain groups if desired via adding `--without <group_name>`.

### Updating packages
`poetry update` or for a specific package, `poetry update transformers`

## Docker

- Check if containers are running: `docker ps`

- Stop a single container: `docker stop <container id from ps>`

- Restart existing containers: from c3po-model-server, run `docker-compose up`

- Delete existing containers: from c3po-model-server, run `docker-compose rm`

- Delete existing containers and all associated volumes: from c3po-model-server, run `docker-compose rm --volumes`

# Other Notes
- We specificy `ENVIRONMENT=local` because the test stage needs the default to be its variables
- For basic CRUD, you can follow this format:
```
from .base import CRUDBase
from app.models.item import Item
from app.schemas.item import ItemCreate, ItemUpdate

item = CRUDBase[Item, ItemCreate, ItemUpdate](Item)
```
- the `env_vars` for `minio` in P1 say secure False, but that is only because the intra-namespace comms between pods get automatically mTLS encrypted via istio, so they keep `http://minio.minio:9000` as the URL inside the namespace.
-`aiohttp` is a subdep of `langchain`, however, do not use it for handling web connections as there are disputed CVEs in that context (disputed as in not official, but it is possible that the risk exists).  See issues here: https://github.com/aio-libs/aiohttp/issues/6772 and `https://github.com/aio-libs/aiohttp/issues/7208`

## Using dgwonub22
Coordinate with other CAITT team members on running the c3po-model-server app; or stopping, starting, and rebuilding the docker containers.

### Browse to MinIO Object Store Console
Chrome:
http://172.25.252.52:9001/login
U: 	miniouser
P: 	minioadmin

Should see MinIO Object Store console.

### Browse to the PostgreSQL Console
Chrome:
http://172.25.252.52:5050
E:	user@test.com
P:	admin

Should see PostgreSQL console.

# Patching a CVE
Usually CVEs can be addressed by easily updating a release, realizing it is a false-positive, or finding a new package.  Inside of P1, if there is a fix and the CVE is low-threat, you can request a whitelist and wait for the official version.  However, if that does not work, you can request that `git` be installed in the pipeline `pip install` runner and use `pip install` with a specific commit addressing the patch.  For example, before 4.30.0 was released, [this transformers CVE](https://nvd.nist.gov/vuln/detail/CVE-2023-2800) could be patched via

`pip install git+https://github.com/huggingface/transformers.git@80ca92470938bbcc348e2d9cf4734c7c25cb1c43#egg=transformers`

and adding

`transformers @ git+https://github.com/huggingface/transformers.git@80ca92470938bbcc348e2d9cf4734c7c25cb1c43`

to the requirements.txt in place of the previous `transformers` installation.

# Knowledge and helpful links
## Tools for this repo
- [Tutorial followed-ish for this repo](https://curiousily.com/posts/deploy-bert-for-sentiment-analysis-as-rest-api-using-pytorch-transformers-by-hugging-face-and-fastapi/)
- [Install conda and tensorflow on Mac M1](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706)
- [`pipenv` with `conda`](https://stackoverflow.com/questions/50546339/pipenv-with-conda)
- [Basics of `pipenv` for application dependency management](https://python.plainenglish.io/getting-started-with-pipenv-d224328799de)
- [Conda and pipenv cheat sheet](https://gist.github.com/ziritrion/8024025672ea92b8bdeb320d6015aa0d)
- [How to use pre-commit framework for git hooks](https://pre-commit.com/index.html)


## Environment updates
- P1 uses pip for environment setup; locally, both poetry and pip are acceptable
- However, ppg-common broke the pre-commit hook that keeps the poetry and pip requirements in sync
- Process for environment updates:
1. Update poetry: $ poetry add package==version
2. Sync with pip: $ ./hooks/output-requirements-txt.sh

## Logs
Logs for the deployed application can be viewed on [ArgoCD](https://argocd-il4.admin.dso.mil/applications?showFavorites=false&proj=&sync=&autoSync=&health=&namespace=&cluster=&labels=)
- click the application
- view in tree, network, or list modes (see icons next to "Log out" at the top right)
- click the ellipsis to the right of the desired pod
- click Logs

## Testing
In general, tensorflow and pytorch use the underlying `unittest` framework that comes stock with Python.  However, FastAPI has a ton of great features through `pytest` that make testing HTTP much, much easier.  Good news is that, for the most part, pytest as the runner will also handle unittest, so we can use the TF or pytorch frameworks with unittest and FastAPI with pytest.  Some articles on this:
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Tensorflow testing](https://theaisummer.com/unit-test-deep-learning/)
- [Pytest handling unittest](https://docs.pytest.org/en/latest/how-to/unittest.html#pytest-features-in-unittest-testcase-subclasses)
- [Mocking in pytest--especially import location](https://changhsinlee.com/pytest-mock/)
- [Better mocking in pytest walkthrough](https://www.toptal.com/python/an-introduction-to-mocking-in-python)
- [Test coverage using `coverage`](https://coverage.readthedocs.io/en/7.2.1/)


## Tools for git
- [Storing Credentials](https://git-scm.com/docs/git-credential-store)...or just type `git config --global credential.helper store`
- [Create a GPG Key](https://confluence.il2.dso.mil/display/CWBI/Create+and+Use+a+GPG+Key) or [GPG Commit Signing](https://confluence.il2.dso.mil/display/afrsba/Setting+up+GPG+for+GitLab+Commit+Signing) or [GitHub Docs](https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key)

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
- [SonarQube for dependency check (data pipeline)](https://sonarqube.il4.dso.mil/dashboard?id=platform-one-products-ai-accel-transformers-c3po-model-server-dependencies)
- [SonarQube for dependency check (model pipeline)](https://sonarqube.il4.dso.mil/dashboard?id=platform-one-products-ai-accel-transformers-c3po-model-gpu-dependencies)
- [SonarQube for SCA & code coverage (data pipeline)](https://sonarqube.il4.dso.mil/dashboard?id=platform-one-products-ai-accel-transformers-c3po-model-server)
- [SonarQube for SCA & code coverage (model pipeline)](https://sonarqube.il4.dso.mil/dashboard?id=platform-one-products-ai-accel-transformers-c3po-model-gpu)
- [False Positive Clearing SQ](https://confluence.il2.dso.mil/display/PUCKBOARD/Sonarqube+False-Positive+Issue+Workflow)
- [Argo page](https://argocd-il4.admin.dso.mil/applications/argocd/p1-il4-mission-staging-transformers-transformers?view=tree&resource=)

### Trufflehog
- [False Positive Clearing for trufflehog](https://confluence.il2.dso.mil/pages/viewpage.action?spaceKey=P1MDOHD&title=TS+-+Trufflehog+-+Stage+Failure)

### K8s configs
- [IL4 mission bootstrap](https://code.il4.dso.mil/platform-one/devops/mission-bootstrap/il4-mission-bootstrap/-/tree/master/integrations/ai-accel/transformers)
- [Deployment manifests](https://code.il4.dso.mil/platform-one/products/ai-accel/transformers/transformers-manifests)
- [IRSA for S3 instructions](https://p1docs.dso.mil/docs/party-bus/mission-devops-mdo/how-tos/s3/#python)

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
