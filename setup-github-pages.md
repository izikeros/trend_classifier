# Setting Up GitHub Pages

This guide explains how to configure GitHub repository settings for automatic documentation deployment.

## Repository Settings

1. Go to your repository on GitHub
2. Click **Settings** (top menu)
3. Click **Pages** (left sidebar under "Code and automation")
4. Under "Build and deployment":
   - **Source**: Select **"GitHub Actions"**
   - No branch selection needed (uses artifact-based deployment)
5. Click **Save**

## First Deployment

1. Push to `main` branch or trigger workflow manually (Actions → Documentation → Run workflow)
2. Wait 2-3 minutes for the action to complete
3. Your site will be live at: `https://izikeros.github.io/trend_classifier/`

## Local Preview

Before pushing, test documentation locally:

```bash
make serve-docs
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) to preview.

## How It Works

The documentation workflow (`.github/workflows/docs.yml`):

1. **Triggers** on push to `main` or manual dispatch
2. **Builds** documentation with `mkdocs build`
3. **Uploads** the `site/` directory as an artifact
4. **Deploys** to GitHub Pages using `deploy-pages@v4`

This modern approach uses GitHub's artifact-based deployment instead of the older `gh-pages` branch method.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 404 on subpages | Ensure `site_url` is set in `mkdocs.yml` |
| Action fails with permission error | Check workflow has `pages: write` permission |
| Site not updating | Check Actions tab for failed workflow runs |
| "There isn't a GitHub Pages site here" | Ensure Pages source is set to "GitHub Actions" |
| Notebooks not rendering | Install `mkdocs-jupyter` in docs dependency group |

## Configuration Files

| File | Purpose |
|------|---------|
| `mkdocs.yml` | MkDocs configuration (theme, plugins, navigation) |
| `.github/workflows/docs.yml` | GitHub Actions workflow for deployment |
| `docs/` | Documentation source files |
| `pyproject.toml` `[dependency-groups.docs]` | Documentation dependencies |
