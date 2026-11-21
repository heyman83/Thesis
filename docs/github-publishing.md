# GitHub publishing guide

Create an empty repository at https://github.com/new using your preferred name and visibility. Leave the README, license, and .gitignore initialization options unchecked because these files already exist locally. Choose a project license separately after reviewing THIRD_PARTY.md.

The local repository is initialized on `main`. Run these commands from the project directory, replacing YOUR-USERNAME and YOUR-REPOSITORY:

```powershell
git status --short
git add .gitignore .gitattributes README.md THIRD_PARTY.md data.example.yaml constraints.txt requirements-lock.txt train.py verify_gpu.py evaluate_thresholds.py
git diff --cached --stat
git commit -m "Add YOLO11n training and evaluation tools"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

If Git requests author details, set `git config user.name "Your Name"` and `git config user.email "your GitHub email or GitHub noreply email"`, then repeat the commit. HTTPS authentication normally opens a browser through Git Credential Manager. GitHub account passwords are not accepted for Git pushes; use the offered browser authentication or a personal access token if prompted.

For subsequent changes, review `git status` and `git diff`, stage the intended files, commit, and run `git push`. Inspect new dataset locations before staging: only the named dataset folders in .gitignore are excluded automatically.

GitHub instructions: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
