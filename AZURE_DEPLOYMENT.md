# Deploying the Regulatory Impact Cockpit to Azure (no GitHub)

This guide deploys the Streamlit cockpit to **Azure App Service (Linux, Python
3.11)** using **ZIP deployment via the Azure CLI**. No GitHub, no Docker, no
Azure DevOps required -- you push the app straight from your Windows laptop.

Total time: ~15 minutes for a first deploy, ~2 minutes for redeploys.

---

## 1. What gets deployed

| Piece                              | Source                                    | Where it runs on Azure                        |
| ---------------------------------- | ----------------------------------------- | --------------------------------------------- |
| Streamlit UI (`app.py`)            | Local repo                                | App Service Linux Python 3.11                 |
| Python dependencies                | `requirements.txt`                        | Installed by Oryx build during ZIP deploy     |
| Startup command                    | `startup.sh`                              | Runs `streamlit run app.py` on `$PORT`        |
| SQLite DB (`data/app.db`)          | Created at runtime                        | Persistent on `/home/site/wwwroot/data/`      |
| Uploads / outputs                  | Created at runtime                        | Persistent on `/home/site/wwwroot/{uploads,outputs}/` |
| Secrets (`API_KEY`) + env vars     | Local `.env` (never uploaded)             | App Service **App Settings** (env vars)       |

**Why App Service `/home` is safe for the DB:** on Linux App Service, `/home`
is a mounted Azure Files share by default and survives restarts, scale-out,
and platform maintenance. That means the SQLite DB, uploads, and outputs
persist without any code change. For heavier concurrent load, migrate to
Azure PostgreSQL and Azure Blob Storage -- see §7.

---

## 2. Prerequisites

1. **Azure subscription** with permission to create Resource Groups (Contributor
   role on the target subscription or an existing RG).
2. **Azure CLI** installed on your Windows laptop:
   [aka.ms/installazurecliwindows](https://aka.ms/installazurecliwindows).
   Verify with:

   ```powershell
   az --version    # should be >= 2.60
   ```

3. **Sign in** (opens a browser):

   ```powershell
   az login
   az account show                       # confirms subscription
   az account set --subscription "<SubscriptionName-or-Id>"
   ```

4. **Working `.env`** in the project root. `deploy_azure.ps1` reads it and
   pushes every key into App Service App Settings. The `.env` file itself is
   never uploaded (it is excluded from the ZIP).

---

## 3. One-command deploy

From `C:\Users\skemidi001\Desktop\Reg_Impact` in PowerShell:

```powershell
.\deploy_azure.ps1 `
    -ResourceGroup   rg-reg-impact `
    -Location        eastus `
    -PlanName        plan-reg-impact `
    -AppName         reg-impact-cockpit-demo `
    -PlanSku         B1
```

Notes:

- **`-AppName` must be globally unique** across `*.azurewebsites.net`. If the
  name is taken you'll get `WebsiteWithGivenNameAlreadyExists` -- pick a new
  suffix.
- **`-Location`**: use a region close to you, e.g. `eastus`, `westeurope`,
  `uksouth`, `centralindia`.
- **`-PlanSku`**:
  - `F1` (Free) -- no Always On, 60 min/day CPU quota. Streamlit websocket may
    drop after idle. **Not recommended.**
  - `B1` (Basic, ~$13/mo) -- smallest tier with Always On. **Recommended for
    demos.**
  - `P1v3` (Premium, ~$85/mo) -- production, VNet integration, autoscale.

What the script does:

1. Creates (or reuses) the Resource Group, Linux App Service Plan, and Web App.
2. Sets the startup command to `bash /home/site/wwwroot/startup.sh`.
3. Enables Always On and HTTPS-only.
4. Reads every non-comment key from `.env` and pushes them as App Settings
   (env vars visible to the app at runtime), plus platform settings
   `SCM_DO_BUILD_DURING_DEPLOYMENT=true`, `ENABLE_ORYX_BUILD=true`,
   `WEBSITES_PORT=8000`, `PYTHONUNBUFFERED=1`.
5. Stages the project (excluding `.env`, `.venv`, `data/`, `uploads/`,
   `outputs/`, `__pycache__/`, IDE folders, DB files, log files), zips it,
   and uploads via `az webapp deploy --type zip`.
6. Prints the live URL like `https://reg-impact-cockpit-demo.azurewebsites.net`.

The first hit takes ~30-60 seconds while Oryx installs `requirements.txt`
in the target container. Subsequent requests are fast.

---

## 4. Redeploying code only

Once the App Settings are in place you can skip re-syncing them:

```powershell
.\deploy_azure.ps1 `
    -ResourceGroup   rg-reg-impact `
    -Location        eastus `
    -PlanName        plan-reg-impact `
    -AppName         reg-impact-cockpit-demo `
    -EnvFile         ""
```

`-EnvFile ""` disables the `.env` → App Settings sync. Only the ZIP is pushed.

---

## 5. Watching logs & debugging

```powershell
# Live stream (Ctrl+C to exit)
az webapp log tail --resource-group rg-reg-impact --name reg-impact-cockpit-demo

# Enable persistent Docker/app logs (do this once)
az webapp log config `
    --resource-group rg-reg-impact `
    --name reg-impact-cockpit-demo `
    --docker-container-logging filesystem `
    --level information

# SSH into the running container to poke around
az webapp ssh --resource-group rg-reg-impact --name reg-impact-cockpit-demo
# Inside SSH:
#   ls -la /home/site/wwwroot/data       # SQLite DB
#   cat /home/site/wwwroot/startup.sh
#   python -c "import streamlit; print(streamlit.__version__)"
```

Common issues:

| Symptom                                              | Cause / Fix                                                                                          |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `Application Error :(` after deploy                  | Startup command wrong or Oryx build failed. Check `az webapp log tail`.                              |
| `ModuleNotFoundError` for something in requirements  | `SCM_DO_BUILD_DURING_DEPLOYMENT` not set to `true`. Re-run deploy or set it in Portal → Configuration. |
| Websocket keeps disconnecting                        | You're on `F1` (no Always On) or App Service is scaled to 0. Upgrade to `B1`.                       |
| SQLite `database is locked`                          | Two writers (e.g. sqlite-web via docker-compose) hitting the same file. Only one writer at a time.  |
| PwC GenAI endpoint unreachable                       | Azure public egress cannot reach `*.pwcinternal.com`. See §6.                                       |

---

## 6. Reaching the PwC internal GenAI endpoint from Azure

Your `.env` points at `https://genai-sharedservice-americas.pwcinternal.com`,
which is a **corporate-internal** endpoint. Azure public egress will not be
able to resolve or reach it. You have three practical options:

1. **Offline mode (fastest for a demo)** -- set `OPENAI_SKIP_API=true` in App
   Settings. The app falls back to the built-in BRD/FRD generator (see
   `services/brd_frd_generator.py`) and skips every GenAI call. Everything
   else -- parsing, questionnaire, scoring, dashboards -- still works.

2. **VNet integration + ExpressRoute / VPN** -- requires a `P0v3+` or `S1+`
   plan. Integrate the Web App into a VNet that peers with (or has a VPN /
   ExpressRoute back to) the PwC corporate network. This is the production
   pattern but needs Network Team involvement.

3. **Egress via corporate proxy** -- if PwC exposes an outbound HTTPS proxy
   reachable from Azure, set `HTTPS_PROXY` / `HTTP_PROXY` App Settings.
   `services/genai_service.py` uses `httpx`, which honours `HTTPS_PROXY`
   automatically.

For a proof-of-concept, **option 1 (offline mode)** is by far the easiest.
Toggle it with:

```powershell
az webapp config appsettings set `
    --resource-group rg-reg-impact `
    --name reg-impact-cockpit-demo `
    --settings OPENAI_SKIP_API=true
```

---

## 7. When to graduate from SQLite + local disk

App Service `/home` is fine up to ~10-20 concurrent users on a `B1` plan.
Beyond that:

- **Database**: swap SQLite for **Azure Database for PostgreSQL Flexible
  Server** (Basic Burstable B1ms is ~$15/mo). Only `services/persistence.py`
  needs to change; the rest of the app uses it as an opaque module.
- **Uploads / outputs**: mount **Azure Files** (or use **Blob Storage**) so
  files survive app deletion and are visible to other tools. Use
  `az webapp config storage-account add` to mount an Azure Files share into
  `/mounted/uploads` and update `UPLOAD_DIR` accordingly.
- **Secrets**: move `API_KEY` from App Settings into **Azure Key Vault** and
  reference it with
  `@Microsoft.KeyVault(SecretUri=https://<vault>.vault.azure.net/secrets/API_KEY/)`.
  The Web App needs a system-assigned managed identity with `get` permission
  on the vault's secrets.

---

## 8. Portal-only path (no CLI)

If you cannot install the Azure CLI:

1. Portal → **Create a resource → Web App**. Runtime stack: **Python 3.11**,
   OS: **Linux**, region: your choice, plan: **B1**.
2. After creation, in the Web App blade:
   - **Configuration → General settings**: Startup Command =
     `bash /home/site/wwwroot/startup.sh`. Enable **Always On** and
     **HTTPS Only**.
   - **Configuration → Application settings**: add every key/value from
     `.env` plus `SCM_DO_BUILD_DURING_DEPLOYMENT=true`,
     `ENABLE_ORYX_BUILD=true`, `WEBSITES_PORT=8000`, `PYTHONUNBUFFERED=1`.
3. Build the ZIP locally:

   ```powershell
   Compress-Archive -Path .\* -DestinationPath .\reg-impact.zip -Force `
       -Exclude .env, .venv, venv, __pycache__, .git, .idea, .vscode, .cursor, data, outputs, uploads, *.db, *.db-journal
   ```

   (Or simply run `deploy_azure.ps1` with a bogus RG name -- it stages and
   zips before it tries to upload, so you can copy the ZIP out of `$env:TEMP`.)

4. In the Web App blade → **Advanced Tools (Kudu) → Go → PowerShell Debug
   Console**. Drag the `reg-impact.zip` onto `/home/site/wwwroot/` and Kudu
   will unpack it and trigger Oryx to install `requirements.txt`.
5. Restart the Web App. Browse to `https://<AppName>.azurewebsites.net`.

---

## 9. Cleaning up

```powershell
# Deletes EVERYTHING in the Resource Group. Use with care.
az group delete --name rg-reg-impact --yes --no-wait
```

---

## 10. Files this deployment relies on

- `startup.sh` -- Streamlit launch command executed by App Service.
- `.deployment` -- tells Kudu to run the Oryx build (`pip install`) on deploy.
- `deploy_azure.ps1` -- end-to-end deploy script (this doc's §3).
- `.env.azure.example` -- reference of every App Setting the app expects,
  minus secrets. Safe to share.
- `requirements.txt` -- Oryx installs these on the App Service container.
- `.streamlit/config.toml` -- theme + server flags; `startup.sh` also passes
  the port explicitly.

Everything else in the repo is application code and travels with the ZIP.
