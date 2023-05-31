# Pre-reqs if you use an ARM-based machine (e.g., Apple M1)--skip if x86 unless you want conda to manage python env, then just skip the tensorflow steps
1. Make sure that you have `conda` installed.  [Recommend this article](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706).

2. Create and activate a new conda environment, e.g., `transformers-api` with python 3.9.
```bash
conda create --name transformers-api python==3.9
conda activate transformers-api
```

3. (Apple silicon only) Install underlying tensorflow dependencies (info pulled from [this article](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706))
```bash
conda install -c apple tensorflow-deps
```

4. Run `which pip` and `which python` to verify path to make sure that your `python` and `pip` binaries are coming from your `conda` virtual environment.  Note that the order in which you install conda vs. pip matters to set virtual env priorities.

# Getting Started Locally
1. Install `poetry`: `pip install poetry` (or use `pipx` [on link here](https://python-poetry.org/docs/1.4#installing-with-pipx) if you prefer isolated envs and you don't have `conda` managing your env)

2. Create and enter the virtual environment: `poetry shell`

3. Install the dependencies `poetry install`

4.  In `c3po-model-server/app/core/env_var`, create a `secrets.env` file and ensure it is on the `.gitignore`.  Add the following for local dev:
```sh
MM_TOKEN="<your_preprod_mattermost_token>"
```

5. Launch postgres, pgadmin, and minio via docker-compose `docker-compose up --build`.

6. Visit `localhost:9001`.  Login with user:`miniouser` and password:`minioadmin`.  This is the minio console.

7. Visit `localhost:5050`.  Login with email:`user@test.com` and password:`admin`.  This is the pgadmin console.  **See notes below for important details**

8. Run the app db init script `./scripts/init.sh`

9. Keeping your docker containers running, start the app in a new terminal (activate your conda env first) with `ENVIRONMENT=local uvicorn app.main:versioned_app --reload`.

10. Open `localhost:8000/v1/docs` and start interacting with swagger!

11. Run tests and get coverage with `ENVIRONMENT=local pytest --cov`, and get html reports for vs code live server (or any server) with `ENVIRONMENT=local pytest --cov --cov-report=html:coverage_re`

12.  You can shut down and your db / minio data will persist via docker volumes.

13. Set up the precommit hook with `pre-commit install`.

# Adding a package
Note: instructions included in [tutorial linked here](https://realpython.com/dependency-management-python-poetry/)
1. Add the package, e.g., `poetry add transformers` or `poetry add transformers --group <group_name>` where `<group_name>` is the dependency group name, e.g., `test` or `dev`.
2. Update the lockfile with `poetry lock` or `poetry lock --no-update` if you don't want poetry to try to update other deps within your existing versioning constraints
3. Install the packages with `poetry install`, exclude certain groups if desired via adding `--without <group_name>`.

# Updating packages
`poetry update` or for a specific package, `poetry update transformers`


# Notes
- You will see that `POSTGRES_SERVER=localhost` in the above steps, however, make sure that you login with hostname `db` in pgAdmin (under the "connection" tab in server properties).  This is because the pgAdmin container is launched in the same docker network as the postgres container, so it uses the service name, whereas launching this app from command line uses port forwarding to localhost.  The user, password, and db name will all be `postgres`, port `5432`.
- We specificy `ENVIRONMENT=local` because the test stage needs the default to be its variables
- For basic CRUD, you can follow this format:
```
from .base import CRUDBase
from app.models.item import Item
from app.schemas.item import ItemCreate, ItemUpdate

item = CRUDBase[Item, ItemCreate, ItemUpdate](Item)
```

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
