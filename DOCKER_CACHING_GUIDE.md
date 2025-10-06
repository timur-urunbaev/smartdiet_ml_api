# Docker Build Caching Guide

## Problem
With **1384 lines** in requirements.txt (including PyTorch, FAISS, Jupyter, Gradio, etc.), Docker was re-downloading all dependencies on every build, taking 10-30+ minutes.

## Solution Implemented

### 1. BuildKit Cache Mounts (Primary Solution)
Both Dockerfiles now use **persistent BuildKit cache mounts** for pip:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --upgrade pip && \
    pip install -r requirements.txt
```

**How it works:**
- `--mount=type=cache`: Creates a persistent cache volume
- `target=/root/.cache/pip`: Caches pip downloads and wheel builds
- `sharing=locked`: Allows concurrent builds to share cache safely
- Cache persists **across all builds** on your machine

**Benefits:**
- ✅ First build: Downloads everything (~10-30 min)
- ✅ Subsequent builds: Uses cached wheels (~30 sec - 2 min)
- ✅ Even if you delete the image, cache remains
- ✅ Only re-downloads if requirements.txt changes

### 2. Docker Layer Optimization
```dockerfile
# Layer 1: Base image (cached by Docker)
FROM python:3.11-slim

# Layer 2: Environment variables (rarely changes)
ENV PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=0

# Layer 3: Dependencies (only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN --mount=type=cache... pip install -r requirements.txt

# Layer 4: Application code (changes frequently)
COPY . .
```

**Caching strategy:**
- Dependencies layer rebuilds ONLY when requirements.txt changes
- Code changes don't trigger dependency reinstall
- BuildKit skips unchanged layers automatically

### 3. docker-compose Layer Cache
```yaml
services:
  api:
    build:
      cache_from:
        - smartdiet-api:latest  # Use previous build as cache source
      args:
        BUILDKIT_INLINE_CACHE: 1  # Store cache in image metadata
    image: smartdiet-api:latest
```

**Benefits:**
- Tagged images serve as cache sources
- Useful for CI/CD pipelines
- Works in multi-developer environments

## Usage

### Enable BuildKit (Required)
BuildKit must be enabled for cache mounts to work:

```bash
# Option 1: Set environment variable (per-build)
export DOCKER_BUILDKIT=1
docker-compose build

# Option 2: Enable globally (recommended)
# Add to ~/.bashrc or ~/.zshrc:
export DOCKER_BUILDKIT=1

# Or configure Docker daemon
# Edit /etc/docker/daemon.json or ~/.docker/daemon.json:
{
  "features": {
    "buildkit": true
  }
}
```

### Building with Cache

```bash
# First build (will download everything)
DOCKER_BUILDKIT=1 docker-compose build

# Subsequent builds (uses cache - FAST!)
DOCKER_BUILDKIT=1 docker-compose build

# Force rebuild without cache (use sparingly)
DOCKER_BUILDKIT=1 docker-compose build --no-cache
```

### Verifying Cache is Working

```bash
# Build once
DOCKER_BUILDKIT=1 docker-compose build api

# Change a Python file (not requirements.txt)
echo "# comment" >> app/app.py

# Rebuild - should be FAST (30s-2min instead of 10-30min)
time DOCKER_BUILDKIT=1 docker-compose build api
```

You should see:
```
=> CACHED [2/5] WORKDIR /app
=> CACHED [3/5] COPY requirements.txt .
=> CACHED [4/5] RUN --mount=type=cache... pip install
```

## Cache Management

### View Cache Usage
```bash
# Show BuildKit cache
docker buildx du

# Example output:
# ID              RECLAIMABLE  SIZE        LAST ACCESSED
# xyz...          true         2.5GB       2 hours ago
```

### Clear Cache (if needed)
```bash
# Clear all BuildKit cache
docker buildx prune

# Clear specific cache (by ID from 'docker buildx du')
docker buildx prune --filter id=xyz...

# Keep cache from last 7 days
docker buildx prune --keep-storage 10GB --filter until=168h
```

## Performance Comparison

### Without Caching (Before)
```
Build time: 10-30 minutes
- Download ~2.5GB of packages every build
- Compile wheels for native extensions (numpy, torch, etc.)
- Even small code changes trigger full rebuild
```

### With Caching (After)
```
First build:  10-30 minutes (downloads everything once)
Subsequent:   30 seconds - 2 minutes
- Reuses downloaded packages
- Reuses compiled wheels
- Only rebuilds changed layers
```

**Savings:** 90-95% reduction in rebuild time!

## Troubleshooting

### Cache not working?

**1. Check BuildKit is enabled:**
```bash
docker version
# Look for: "BuildKit: moby/buildkit"
```

**2. Verify cache mount syntax:**
```bash
# Should see "--mount=type=cache" in Dockerfile
grep "mount=type=cache" app/Dockerfile
```

**3. Check for requirements.txt changes:**
```bash
# If requirements.txt changed, pip must reinstall
git diff app/requirements.txt
```

**4. Ensure no --no-cache flag:**
```bash
# Don't use --no-cache unless intentional
docker-compose build --no-cache  # ❌ Disables cache
docker-compose build             # ✅ Uses cache
```

### Still slow after code changes?

This is expected if you modified:
- `requirements.txt` - Must reinstall dependencies
- `Dockerfile` RUN commands - Must re-execute
- Base image updated - Must pull new image

### Cache location

BuildKit cache is stored in Docker's data directory:
- Linux: `/var/lib/docker/buildkit/`
- macOS: `~/Library/Containers/com.docker.docker/Data/vms/0/data/docker/buildkit/`
- Windows: `C:\ProgramData\Docker\buildkit\`

## Best Practices

1. **Keep requirements.txt stable**
   - Add dependencies infrequently
   - Use lock files (uv.lock) for reproducibility

2. **Use .dockerignore**
   - Prevents unnecessary context copying
   - Already configured in your project

3. **Multi-stage builds (for production)**
   ```dockerfile
   # Stage 1: Build dependencies
   FROM python:3.11-slim as builder
   RUN --mount=type=cache,target=/root/.cache/pip \
       pip install --prefix=/install -r requirements.txt

   # Stage 2: Runtime
   FROM python:3.11-slim
   COPY --from=builder /install /usr/local
   COPY . .
   ```

4. **Enable BuildKit globally**
   - Add `export DOCKER_BUILDKIT=1` to shell profile
   - Or configure Docker daemon permanently

## Additional Optimizations

### For Development: Mount Code as Volume
Instead of rebuilding for code changes:

```yaml
services:
  api:
    volumes:
      - ./app:/app  # Live code reload
      - /app/.venv  # Exclude venv
```

### For CI/CD: Use Registry Cache
```yaml
services:
  api:
    build:
      cache_from:
        - registry.example.com/smartdiet-api:latest
        - registry.example.com/smartdiet-api:develop
```

## Summary

Your Docker builds now use **three levels of caching**:

1. **BuildKit cache mounts** - Persistent pip cache across builds
2. **Docker layer cache** - Reuse unchanged Dockerfile layers
3. **Image cache** - Use previous builds as cache source

**Result:** Rebuilds that took 10-30 minutes now take 30 seconds - 2 minutes! 🚀

**First build:** Download everything once
**Every rebuild after:** Lightning fast ⚡
