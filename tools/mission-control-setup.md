# Mission Control Setup

Mission Control is included as a Git submodule at `tools/mission-control`.

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:Farrells-LLC/open-claw-for-archimedes.git
```

If the repo is already cloned:

```bash
git submodule update --init --recursive
```

Local setup:

```bash
cd tools/mission-control
corepack pnpm install --frozen-lockfile
cp .env.example .env
```

Set these values in `.env` for a local OpenClaw gateway:

```bash
OPENCLAW_HOME=/home/dash/.openclaw
OPENCLAW_CONFIG_PATH=/home/dash/.openclaw/openclaw.json
OPENCLAW_GATEWAY_HOST=127.0.0.1
OPENCLAW_GATEWAY_PORT=18789
NEXT_PUBLIC_GATEWAY_PORT=18789
MC_DEFAULT_GATEWAY_NAME=primary
MC_COORDINATOR_AGENT=main
NEXT_PUBLIC_COORDINATOR_AGENT=main
```

Run for LAN preview:

```bash
PORT=3000 ./node_modules/.bin/next dev --hostname 0.0.0.0 --port 3000
```

Then open:

```text
http://<host-lan-ip>:3000/setup
```

Do not commit `.env`, `.data/`, or `node_modules/`.
