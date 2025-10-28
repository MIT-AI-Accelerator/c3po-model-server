# If you want to use conda, not required
1. Make sure that you have `conda` installed.  [Recommend this article if on Mac, just do through step 2](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706).

2. Create and activate a new conda environment, e.g., `transformers-api` with python 3.12.
```bash
conda create --name transformers-api python=3.12
conda activate transformers-api
```

3. Run `which pip` and `which python` to verify path to make sure that your `python` and `pip` binaries are coming from your `conda` virtual environment.  Note that the order in which you install conda vs. pip matters to set virtual env priorities.

# Getting Started Locally (Start here if not using conda, just make sure you have the right version of python and pip installed)

This procedure runs the c3po-model-server without connecting to Mattermost.
It builds and runs the containers, initializes the data, and runs pytests.

VITAL: Do not use the Mattermost Token.
Use the integration environment


The containers used by c3po-model-server have been built using either:

•	docker-compose up --build
•	docker compose build and docker compose up

WARNING: Support for   docker-compose, as opposed to docker compose, officially ended in June 2023.

ERROR: Sometimes the unsupported version fails.


1. Install `poetry` version 1.8.5: `pip install poetry==1.8.5` (or use `pipx` [on link here](https://python-poetry.org/docs/1.4#installing-with-pipx) if you prefer isolated envs and you don't have `conda` managing your env)

2. Create and enter the virtual environment: `poetry shell`. Note: if you use conda, this step may not be necessary.

3. Install the dependencies `poetry install`

4. Optionally, build postgres, pgadmin, and minio via `docker compose build`.

5. Launch postgres, pgadmin, and minio via docker-compose `docker compose up`.
. 
6. Visit `localhost:9001`.  Login with user:`miniouser` and password:`minioadmin`.  This is the minio console.

7. Visit `localhost:5050`.  Login with email:`user@test.com` and password:`admin`.  This is the pgadmin console.  **See notes below for important details**

8. Run the app db init script `ENVIRONMENT=integration ./scripts/init.sh`

9. Keeping your docker containers running, start the app in a new terminal (activate your conda env first) with `ENVIRONMENT=integration uvicorn app.main:versioned_app --reload`.

10. Open `localhost:8000/v1/docs` and start interacting with swagger!

11. Run tests and get coverage with `ENVIRONMENT=integration pytest --cov`, and get html reports for vs code live server (or any server) with `ENVIRONMENT=integration pytest --cov --cov-report=html:coverage_re`

12.  You can shut down and your db / minio data will persist via docker volumes.


# Getting Started with Mattermost

1.  In `c3po-model-server/app/core/env_var`, create a `secrets.env` file and ensure it is on the `.gitignore`.  Add the following for local dev:
```sh
MM_TOKEN="<your_preprod_mattermost_token>"
```
2. Run the app db init script `ENVIRONMENT=local ./scripts/init.sh`

3. Keeping your docker containers running, start the app in a new terminal (activate your conda env first) with `ENVIRONMENT=local uvicorn app.main:versioned_app --reload`.

4. Run tests and get coverage with `ENVIRONMENT=local pytest --cov`, and get html reports for vs code live server (or any server) with `ENVIRONMENT=local pytest --cov --cov-report=html:coverage_re`

# Using VM dgwonub22
NOTE: “Currently” in this section means Oct. 2025.

PPG runs on macOS and on Linux.

CAITT members whose primary computer runs Windows use the VM dgwonub22.
They do not use WSL2.

Other CAITT members may use the VM in addition to their personal laptops.

The VM is currently running Ubuntu 22.04.5 LTS.

## Sudo privileges:

Because the dgwonub22 installation of Docker requires Root privileges, you must have sudo privileges to build, start, stop, and monitor PPG containers. See IT if necessary.

## Collisions on VM dgwonub22:

Currently the following CAITT members have access to VM dgwonub22:

da32459	Dan Gwon
em23761	Emilie Cowen
jo25802	John Holodnak
pa27879	Paul Gibby
ds14673	Scott Briere	IT
ju21882	Justin O’Brien	IT
ve22990	Vern Rivet		IT

WARNING: More than one of the above non-IT people may attempt to use the c3po-model-server simultaneously, possibly stopping, starting, and rebuilding the containers.

Currently there is no formal system for coordinating this.

Note: An informal way has been to disable sudo privileges for all but one user, thus preventing access to the containers. If you suddenly encounter this, contact Emilie.

## Disk space on VM dgwonub22:

VM dgwonub22 has disk space limits that can not be increased.

Currently, disk usage is not monitored.

You may get error messages when 
•	Building or starting the containers
•	Running the app db init script, ./scripts/init.sh.
•	Running the pytests

BEST PRACTICE: When running the app db script, always read the displayed event log and grep for “space”. It is easy to overlook this.

Examples:
•	db-1  | 2025-10-22 17:49:06.144 UTC [1] FATAL:  could not write lock file "postmaster.pid": No space left on device
•	pgadmin-1  | OSError: [Errno 28] No space left on device

Also, the effects can be progressive. If you start seeing inexplicable errors, check your disk space.

If you are out of disk space, contact Emilie, who will contact IT.

## Other Apps on VM dgwonub22:

Currently the VM hosts:

•	PPG c3po-model-server
•	UDL Data Collection

Currently there are no known dependencies between these apps.

WARNING, they have different environments, conda or otherwise, so be careful if jumping back and forth between them.

## Logging in and setting environments:

On your computer, enter:
$ ssh LincolnId@dgwonub22.llan.ll.mit.edu

Use your Lincoln password.

If you are using conda,
If you have not set your shell to automatically load conda:
Enter:
$ eval "$(/home/jo20812/miniconda3/bin/conda shell.bash hook)"

Then enter:
$ conda activate transformers-api

## Browsing in to VM dgwonub22:

PPG uses Chrome for:

•	The minIO Object Store console
•	The PostgreSQL console
•	Using SWAGGER to access the c3po-model-server API

Mac users use localhost to access the server running on their Mac
VM users must use an IP address to access the server running on the VM.

## Browse to MinIO Object Store Console
Chrome:
http://172.25.252.52:9001/login
U: 	miniouser
P: 	minioadmin

Should see MinIO Object Store console.


## Browse to the PostgreSQL Console
Chrome:
http://172.25.252.52:5050
E:	user@test.com
P:	admin

Should see PostgreSQL console.

## Browse to the SWAGGER UI

You need to expose the server to the external network by giving the host parameter

Launch the server app with:
ENVIRONMENT=integration uvicorn app.main:versioned_app --reload –host 0.0.0.0

Then you can access the swagger UI at http://dgwonub22:llan.ll.mit.edu:8000/v1/docs



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
- the `env_vars` for `minio` in P1 say secure False, but that is only because the intra-namespace comms between pods get automatically mTLS encrypted via istio, so they keep `http://minio.minio:9000` as the URL inside the namespace.
-`aiohttp` is a subdep of `langchain`, however, do not use it for handling web connections as there are disputed CVEs in that context (disputed as in not official, but it is possible that the risk exists).  See issues here: https://github.com/aio-libs/aiohttp/issues/6772 and `https://github.com/aio-libs/aiohttp/issues/7208`

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
