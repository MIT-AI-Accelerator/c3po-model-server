#! /usr/bin/env bash
# run from root of project

# Let the DB start
# python /app/app/backend_pre_start.py

# Run migrations
# alembic upgrade head

# Create initial data in DB
ENVIRONMENT=production python -m app.initial_data --toggle-migration
