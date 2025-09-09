# GitHub Actions for CoexistAI Docker Builds

This document describes the GitHub Actions workflows for building, testing, and deploying multi-architecture Docker images for CoexistAI.

## Overview

We have implemented a comprehensive CI/CD pipeline using GitHub Actions that:
- Builds multi-architecture Docker images (AMD64, ARM64, ARMv7)
- Pushes images to GitHub Container Registry (GHCR)
- Runs security scans and generates SBOMs
- Tests images before deployment
- Manages releases with semantic versioning

## Workflows

### 1. Main Build Workflow (`docker-build-multiarch.yml`)

**Purpose**: Primary workflow for building and pushing multi-arch Docker images

**Triggers**:
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main` or `master`
- Git tags matching `v*` pattern
- Manual dispatch

**Features**:
- Multi-platform builds (linux/amd64, linux/arm64, linux/arm/v7)
- Intelligent tagging strategy
- Security scanning with Trivy
- SBOM generation
- Automated testing
- Build caching for performance

**Usage**:
```bash
# Automatic on push/PR
git push origin main

# Manual trigger with custom tag
# Go to Actions tab → Run workflow → Enter custom tag
```

### 2. Local Testing Workflow (`docker-test-local.yml`)

**Purpose**: Quick testing of Docker builds without pushing

**Triggers**:
- Pull requests affecting Docker files
- Manual dispatch

**Features**:
- Fast AMD64-only builds
- Local testing without registry push
- Security scanning
- Image size analysis
- Layer analysis

**Usage**:
```bash
# Automatic on PR
# Or manual via Actions tab
```

### 3. Release Workflow (`release.yml`)

**Purpose**: Manage semantic versioning and create releases

**Triggers**:
- Git tags matching `v*` pattern
- Manual dispatch with version input

**Features**:
- Semantic versioning support
- GitHub release creation
- Changelog generation
- Multi-platform image building
- Docker Hub integration (optional)

**Usage**:
```bash
# Create and push a tag
git tag v1.0.0
git push origin v1.0.0

# Or manual via Actions tab
```

## Configuration

### Repository Settings

1. **Package Settings**:
   - Go to Settings → Packages
   - Ensure "Inherit access from source repository" is enabled
   - Set visibility to public (for open source) or private

2. **Actions Permissions**:
   - Settings → Actions → General
   - Enable "Read and write permissions"
   - Enable "Allow GitHub Actions to create and approve pull requests"

### Required Secrets

The workflows use `GITHUB_TOKEN` which is automatically provided. No additional secrets are required for basic functionality.

**Optional Secrets** (for extended features):
- `DOCKERHUB_USERNAME` - For Docker Hub sync
- `DOCKERHUB_PASSWORD` - For Docker Hub sync

## Image Tags

The workflows generate intelligent tags:

### Branch Builds
- `latest` - for main/master branch
- `develop` - for develop branch
- `feature-branch` - for feature branches

### Pull Request Builds
- `pr-123` - for PR #123

### Release Builds
- `v1.0.0` - full version
- `v1.0` - major.minor version
- `v1` - major version only
- `latest` - always points to latest release

### Commit-based Tags
- `main-abc123` - commit SHA with branch prefix

## Security Features

### Vulnerability Scanning
- **Tool**: Trivy (Aqua Security)
- **Frequency**: Every build
- **Reports**: GitHub Security tab
- **Fail conditions**: Critical and High severity vulnerabilities

### SBOM Generation
- **Format**: SPDX JSON
- **Content**: All dependencies and layers
- **Storage**: Workflow artifacts
- **Compliance**: Supply chain security

### Image Signing (Optional)
To enable image signing, add these secrets:
- `COSIGN_PRIVATE_KEY`
- `COSIGN_PASSWORD`

## Performance Optimization

### Build Caching
```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

### Multi-stage Builds
The Dockerfile uses multi-stage builds to minimize final image size:
- Builder stage: Compiles dependencies
- Runtime stage: Minimal runtime environment

### Parallel Processing
- Multi-platform builds run in parallel
- Testing runs independently
- Security scans run concurrently

## Monitoring and Debugging

### Workflow Status
- Check Actions tab in repository
- Green checkmarks = success
- Red X = failure with logs

### Image Information
```bash
# List available images
docker pull ghcr.io/username/coexistai:latest

# Check image details
docker image inspect ghcr.io/username/coexistai:latest

# Run locally
docker run -p 8000:8000 ghcr.io/username/coexistai:latest
```

### Debugging Failed Builds
1. Check workflow logs in GitHub Actions
2. Enable debug logging:
   ```yaml
   env:
     ACTIONS_STEP_DEBUG: true
   ```
3. Test locally with act: https://github.com/nektos/act

## Cost Management

### GitHub Actions Free Tier
- **Public repositories**: Unlimited minutes
- **Private repositories**: 2,000 minutes/month
- **Multi-arch builds**: ~10-15 minutes per build

### Optimization Tips
1. Use build caching
2. Limit platform builds for PRs
3. Skip unnecessary workflows with path filters
4. Use self-hosted runners for heavy builds (optional)

## Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check Dockerfile syntax
   docker build -t test .
   
   # Check dependencies
   pip install -r config/requirements.txt
   ```

2. **Permission Errors**
   - Ensure `GITHUB_TOKEN` has write permissions
   - Check package settings in repository

3. **Platform Issues**
   - Some Python packages don't support ARM
   - Check for platform-specific dependencies
   - Use QEMU for cross-platform builds

4. **Cache Issues**
   ```bash
   # Clear GitHub Actions cache
   # Go to Actions → Caches → Delete relevant caches
   ```

### Getting Help
1. Check workflow logs for specific errors
2. Review GitHub Actions documentation
3. Open an issue in the repository
4. Check GitHub status page for service issues

## Best Practices

### Commit Messages
Use conventional commits for better changelog generation:
```bash
git commit -m "feat: add new search functionality"
git commit -m "fix: resolve memory leak in embeddings"
git commit -m "docs: update API documentation"
```

### Branch Strategy
- `main`/`master`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `hotfix/*`: Critical fixes

### Release Process
1. Create release branch from `main`
2. Update version numbers
3. Test thoroughly
4. Create and push tag
5. Monitor deployment

## Advanced Usage

### Custom Build Arguments
Modify the workflow to add build arguments:
```yaml
build-args: |
  CUSTOM_VERSION=${{ steps.version.outputs.VERSION }}
  BUILD_DATE=${{ steps.date.outputs.DATE }}
```

### Multiple Registries
Push to multiple registries simultaneously:
```yaml
- name: Log in to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_PASSWORD }}

- name: Build and push to multiple registries
  uses: docker/build-push-action@v5
  with:
    tags: |
      ghcr.io/${{ github.repository }}:latest
      docker.io/${{ secrets.DOCKERHUB_USERNAME }}/coexistai:latest
```

### Conditional Builds
Add conditions to control when workflows run:
```yaml
if: |
  github.event_name == 'push' && 
  contains(github.event.head_commit.message, '[build]')
```

This comprehensive setup provides a robust CI/CD pipeline for CoexistAI with multi-architecture support, security scanning, and automated testing - all using GitHub's free public runners.