# GitHub Actions Workflow for Multi-Arch Docker Builds

This workflow automatically builds and pushes multi-architecture Docker images to the GitHub Container Registry (GHCR).

## Features

- **Multi-platform support**: Builds for `linux/amd64` and `linux/arm64`
- **Automated tagging**: Intelligent tagging based on branches, PRs, and semantic versioning
- **Security scanning**: Integrated vulnerability scanning with Trivy
- **SBOM generation**: Generates Software Bill of Materials for compliance
- **Caching**: Uses GitHub Actions cache for faster builds
- **Testing**: Automated testing of built images
- **Public runners**: Uses GitHub's free public runners (no local infrastructure needed)

## Triggers

The workflow runs automatically on:
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main` or `master` branches
- Git tags matching `v*` pattern (e.g., `v1.0.0`)
- Manual dispatch via workflow_dispatch

## Usage

### Automatic Builds
Simply push your code to the repository. The workflow will:
1. Build multi-arch images
2. Push to GHCR
3. Run security scans
4. Test the images
5. Generate SBOM

### Manual Build with Custom Tag
1. Go to Actions tab in your repository
2. Select "Build and Push Multi-Arch Docker Image"
3. Click "Run workflow"
4. Enter a custom tag (optional)
5. Click "Run workflow"

### Image Tags

The workflow generates the following tags:
- `latest` - for main/master branch builds
- `branch-name` - for feature branch builds
- `pr-#` - for pull request builds
- `v1.0.0` - for semantic version tags
- `v1.0`, `v1` - for major.minor and major version tags
- `sha-abc123` - for commit SHA tags
- Custom tags via manual dispatch

## Configuration

### Required Secrets
- `GITHUB_TOKEN` - Automatically provided by GitHub (no setup needed)

### Environment Variables
- `REGISTRY`: Set to `ghcr.io` (GitHub Container Registry)
- `IMAGE_NAME`: Automatically set to your repository name

### Build Arguments
The workflow supports build arguments that can be passed to the Dockerfile:
- `BUILDKIT_INLINE_CACHE=1` - Enables BuildKit inline caching

## Security Features

### Vulnerability Scanning
- Uses Trivy to scan for vulnerabilities
- Results are uploaded to GitHub Security tab
- SARIF format for GitHub integration

### SBOM Generation
- Generates Software Bill of Materials in SPDX format
- Artifacts are available for download after workflow completion

## Testing

The workflow includes automated testing:
- Tests both `linux/amd64` and `linux/arm64` platforms
- Verifies health endpoint (`/health`)
- Tests root endpoint (`/`)
- Runs container for 30 seconds to ensure stability

## Performance Optimizations

### Build Caching
- Uses GitHub Actions cache for Docker layers
- BuildKit inline caching enabled
- Multi-stage build optimization in Dockerfile

### Parallel Builds
- Multi-platform builds run in parallel
- Testing runs independently after successful build

## Monitoring

### Workflow Status
- Check the Actions tab in your repository
- Green checkmarks indicate successful builds
- Red X indicates failures with detailed logs

### Image Availability
- Images are available at: `ghcr.io/${{ github.repository }}`
- Example: `ghcr.io/username/coexistai:latest`

### Security Reports
- Vulnerability reports in GitHub Security tab
- SBOM artifacts in workflow run artifacts

## Troubleshooting

### Common Issues

1. **Build failures**: Check Dockerfile syntax and dependencies
2. **Permission errors**: Ensure `GITHUB_TOKEN` has proper permissions
3. **Platform issues**: Some dependencies may not support all architectures
4. **Cache issues**: Clear cache by re-running workflow

### Debug Mode
To enable debug logging:
1. Set repository secret: `ACTIONS_STEP_DEBUG=true`
2. Re-run the workflow for detailed logs

## Cost Considerations

This workflow uses GitHub's free public runners:
- **Free tier**: 2,000 minutes/month for public repositories
- **Multi-arch builds**: Take longer but run on free infrastructure
- **Caching**: Reduces build time and costs

## Customization

### Adding More Platforms
Edit the `platforms` line in the workflow:
```yaml
platforms: linux/amd64,linux/arm64,linux/arm/v7,linux/s390x
```

### Custom Build Arguments
Add build arguments in the workflow:
```yaml
build-args: |
  CUSTOM_ARG=value
  ANOTHER_ARG=value
```

### Additional Testing
Extend the test job with custom tests:
```yaml
- name: Custom test
  run: |
    docker run --rm your-custom-test-command
```

## Support

For issues or questions:
1. Check the workflow logs in GitHub Actions
2. Review the troubleshooting section above
3. Open an issue in the repository