#!/bin/bash
# Set the environment variable DATASET_SERVER_ADDR to the current host's IP address
export DATABASE_SERVER_IP="$(hostname -i)"

# Launch the FastAPI queue server (database_server.py)
python3 database_server.py