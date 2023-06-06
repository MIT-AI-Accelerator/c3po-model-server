#! /usr/bin/env bash

if [ "$ENVIRONMENT" = "local" ] || [ "$ENVIRONMENT" = "test" ] || [ -z $ENVIRONMENT ]
then
    echo "running local script"
    ./scripts/init-local.sh
elif [ "$ENVIRONMENT" = "development" ]
then
    echo "running development script"
    ./scripts/init-development.sh
elif [ "$ENVIRONMENT" = "staging" ]
then
    echo "running staging script"
    ./scripts/init-staging.sh
elif [ "$ENVIRONMENT" = "production" ]
then
    echo "running production script"
    ./scripts/init-production.sh
else
    echo "Invalid environment"
fi
echo "init script finished"
