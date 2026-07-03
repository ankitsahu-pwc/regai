#!/usr/bin/env bash
# Azure App Service (Linux, Python) startup command for the Streamlit cockpit.
#
# App Service exposes the public port through the $PORT environment variable
# (usually 8000). We bind Streamlit to 0.0.0.0 on that port so the platform's
# front-end reverse proxy can route requests to the container.
#
# Notes:
# * We rely on Oryx (SCM_DO_BUILD_DURING_DEPLOYMENT=true) to install
#   requirements.txt during deploy, so no pip install here.
# * /home is persistent on App Service Linux, and the app is deployed to
#   /home/site/wwwroot, so ./data (SQLite), ./uploads, and ./outputs survive
#   restarts without any code change.

set -euo pipefail

cd /home/site/wwwroot

exec python -m streamlit run app.py \
    --server.port="${PORT:-8000}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
