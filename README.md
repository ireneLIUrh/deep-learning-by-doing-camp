# deep-learning-by-doing-camp
Learning deep learning by doing: notes, code, logs, and relative materials.



## Daily Workflow

This is the standard daily workflow for maintaining this learning project.

### 1. Enter the project

```bash
cd /mnt/backup02/rhliu/projects/deep-learning-by-doing-camp
conda activate dlcamp
```

Check the current status:

```bash
pwd
git status
```

### 2. Sync the latest version

Always pull the latest version before editing:

```bash
git pull origin main
```

### 3. Study, write, and code

Work on one small topic each day.

Typical files to update:

```text
docs/       # notes, logs, tutorials
scripts/    # runnable scripts
notebooks/  # exploratory notebooks
src/        # reusable code
```

Recommended daily output:

```text
one concept
→ one note
→ one small code example
→ one commit
→ one push
→ one deployment
```

### 4. Build and preview the documentation site

First check whether the site can be built:

```bash
mkdocs build
```

Then preview locally from the remote server:

```bash
mkdocs serve --dev-addr=0.0.0.0:8077
```

Use MobaXterm local port forwarding:

```text
Local port: 8077
Remote server: localhost
Remote port: 8077
SSH server: your server address
SSH login: rhliu
```

Open in the local browser:

```text
http://localhost:8077
```

After checking the website, stop the preview server with:

```text
Ctrl + C
```

### 5. Check Git status

```bash
git status
```

Before committing, make sure large or generated files are not included, such as:

```text
site/
.venv/
data/raw/
data/processed/
large result files
```

### 6. Commit changes

Because the server uses an old Git version, use `git add -A` instead of `git add .`.

```bash
git add -A
git commit -m "Update learning notes"
```

Use clear commit messages, for example:

```bash
git commit -m "Add notes on matrix multiplication"
git commit -m "Add pure Python matrix multiplication script"
git commit -m "Update week1 learning log"
git commit -m "Fix MkDocs navigation"
```

### 7. Push to GitHub

```bash
git push origin main
```

The repository uses SSH remote, so the remote should look like:

```bash
git remote -v
```

Expected:

```text
origin  git@github.com:ireneLIUrh/deep-learning-by-doing-camp.git (fetch)
origin  git@github.com:ireneLIUrh/deep-learning-by-doing-camp.git (push)
```

### 8. Deploy to GitHub Pages

If `docs/` or `mkdocs.yml` was changed, deploy the website:

```bash
mkdocs gh-deploy --force
```

Then check the published site:

```text
https://ireneliurh.github.io/deep-learning-by-doing-camp/
```

### Full daily command template

```bash
cd /mnt/backup02/rhliu/projects/deep-learning-by-doing-camp
conda activate dlcamp

git pull origin main

# Edit docs/, scripts/, notebooks/, or src/
# Recommended editors: VS Code Remote SSH, github.dev, Typora, or local VS Code

mkdocs build
mkdocs serve --dev-addr=0.0.0.0:8077

# Check http://localhost:8077 in the local browser
# Press Ctrl + C after preview

git status
git add -A
git commit -m "Update learning notes"
git push origin main

mkdocs gh-deploy --force
```

### Weekly recap

At the end of each week, update the weekly log with:

```markdown
## Weekly Recap

### What did I finish this week?

### What concepts are still unclear?

### What code should be improved?

### What is the plan for next week?

### Important commits
```

Then commit and deploy:

```bash
git add -A
git commit -m "Add weekly recap"
git push origin main
mkdocs gh-deploy --force

```
