# Releasing

The repository lives at <https://github.com/bartsikorski/mandelscope>.

## Publishing for the first time (Windows, PowerShell)

1. Install Git once: `winget install --id Git.Git -e`, then open a new PowerShell window. Set your name and email for commits:
   ```
   git config --global user.name "bartsikorski"
   git config --global user.email "you@example.com"   # the email of your GitHub account, or its noreply address
   ```
   The private noreply address is under GitHub → Settings → Emails.
2. On GitHub, create an **empty** repository: <https://github.com/new>, owner `bartsikorski`, name `mandelscope`, Public. Don't add a README, .gitignore or licence; they are already here.
3. In the project folder:
   ```
   cd "C:\path\to\mandelscope"      # the folder with README.md
   if (Test-Path _github) { Rename-Item _github .github }
   git init -b main
   git add .
   git status
   ```
   `git status` must not list `.venv`, `footage`, `dist`, `build` or `software`; `.gitignore` keeps them out.
4. Commit and push:
   ```
   git commit -m "Mandelscope 1.0.0"
   git remote add origin https://github.com/bartsikorski/mandelscope.git
   git push -u origin main
   ```
   The first push opens a browser window to sign in to GitHub.
5. Open <https://github.com/bartsikorski/mandelscope/actions>. The **CI** workflow runs two jobs: `test` (Linux) and `windows` (CPU mode). Wait until both are green.
6. Make the release:
   ```
   git tag v1.0.0
   git push origin v1.0.0
   ```
   The **Release** workflow builds `Mandelscope.exe` with Mesa for CPU mode, checks that it starts, and publishes `Mandelscope-v1.0.0-windows-x64.zip` on <https://github.com/bartsikorski/mandelscope/releases>.
7. Optional, on the repository page: *About* (gear icon) → description and topics (`mandelbrot`, `fractal`, `deep-zoom`, `gpu`, `opengl`, `python`). Settings → General → *Social preview* → upload `docs/images/seahorse_valley.jpg`.

## Later releases

1. Update `version` in `pyproject.toml` and `__version__` in `mandelscope/__init__.py`, and add a section to `CHANGELOG.md`.
2. Run the checks:
   ```
   uv run ruff check .
   uv run pytest -q
   ```
   Then `run.bat bench --experiments --bla` on a real GPU and note anything unusual in the changelog.
3. Commit, push, and tag once CI is green:
   ```
   git commit -am "Mandelscope 1.1.0"
   git push
   git tag v1.1.0
   git push origin v1.1.0
   ```
4. The **Release** workflow attaches `Mandelscope-v1.1.0-windows-x64.zip` to a new GitHub release with generated notes. Edit the notes on GitHub if you like, for example by pasting the changelog section.

The **CI** workflow runs the lint and the full test suite on every push and pull request: on Linux with Mesa's software OpenGL on a virtual display, and on Windows in CPU mode. Neither needs a GPU.
