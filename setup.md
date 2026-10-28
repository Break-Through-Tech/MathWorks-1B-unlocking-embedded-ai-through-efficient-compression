# Environment Setup Guide

This guide walks through setting up the shared project environment for Mathworks 1B.
Follow the section for your operating system in each step.

The environment is defined in `environment-full.yml` in the repo root. Do not
install packages individually outside of this file — see **Section 8** for
how to add something new.

---

## 1. Install Miniforge (Anaconda + Mamba, in one installer)

We use **Miniforge** rather than the full Anaconda Distribution. It installs
the same `conda` you'd get from Anaconda, defaults to the free conda-forge
channel, and ships `mamba` (a much faster installer) out of the box — so you
only need to install one thing instead of two.

> If your program specifically requires the full Anaconda Distribution
> (e.g. Anaconda Navigator's GUI), install that instead from
> https://www.anaconda.com/download and then skip to Section 2 to add mamba
> on top of it.

### Mac

1. Go to https://github.com/conda-forge/miniforge/releases/latest
2. Download the installer matching your chip:
   - Apple Silicon (M1/M2/M3/M4): `Miniforge3-MacOSX-arm64.sh`
   - Intel Mac: `Miniforge3-MacOSX-x86_64.sh`
3. Open Terminal, navigate to your Downloads folder, and run:
   ```bash
   bash Miniforge3-MacOSX-arm64.sh
   ```
   (substitute the `x86_64` filename if you're on an Intel Mac)
4. Accept the license, accept the default install location, and when asked
   "Do you wish to update your shell profile to automatically initialize
   conda?" — answer **yes**.
5. Close and reopen Terminal.
6. Verify it worked:
   ```bash
   conda --version
   mamba --version
   ```

### Windows

1. Go to https://github.com/conda-forge/miniforge/releases/latest
2. Download `Miniforge3-Windows-x86_64.exe`
3. Run the installer.
   - Choose **"Just Me"** (not "All Users") unless you have a specific reason not to.
   - On the "Advanced Options" screen, it's fine to leave "Add Miniforge3 to
     my PATH environment variable" **unchecked** — the installer sets up its
     own "Miniforge Prompt" shortcut instead, which is the recommended way to
     use it on Windows.
4. Open the **Miniforge Prompt** from the Start Menu (not a regular Command
   Prompt or PowerShell window yet).
5. Verify it worked:
   ```
   conda --version
   mamba --version
   ```

---

## 2. Initialize Mamba for your shell

Miniforge sets up its own prompt automatically, but if you want to use
`mamba activate` from your regular terminal (Terminal.app, iTerm, PowerShell,
Windows Terminal), you need to run a one-time hook setup. If you always plan
to use the Miniforge Prompt / the terminal that opened by default after
install, you can skip this section.

### Mac (zsh — the default shell on modern macOS)

```bash
mamba shell init --shell zsh --root-prefix=~/.local/share/mamba
exec zsh
```

The first command writes a hook into your `~/.zshrc`; the second reloads your
shell so it takes effect immediately.

If you're on an older Mac still using bash, substitute `--shell bash` above.

### Windows (PowerShell)

```powershell
mamba shell init --shell powershell --root-prefix=~/.local/share/mamba
```

Then close and reopen PowerShell, or reload without restarting:
```powershell
. $PROFILE
```

**Common snag:** PowerShell may refuse to run the activation script with an
error like *"running scripts is disabled on this system."* Fix it (as
yourself, not as admin) with:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows (cmd.exe)

```
mamba shell init --shell cmd.exe --root-prefix=~/.local/share/mamba
```
Close and reopen the terminal window afterward (cmd.exe can't reload its
profile in-place the way PowerShell can).

---

## 3. Clone the repository

Same on both platforms, assuming Git is installed
(https://git-scm.com/downloads if not):

```bash
git clone <repo-url>
cd <repo-folder>
```

---

## 4. Create the environment

From inside the repo folder, on **either OS**:

```bash
mamba env create -f environment-full.yml
```

This reads `environment-full.yml`, downloads Python 3.12 and every listed
package, and builds an environment named `bearing-fault-diagnosis`. This step
installs several GB of packages (TensorFlow and PyTorch both included) so it
can take a while — let it run to completion and watch for red error text,
which would mean something failed partway through.

---

## 5. Activate the environment

Same command on both platforms, every time you start work:

```bash
mamba activate bearing-fault-diagnosis
```

**If you see an error like:**
```
Cannot activate, prefix does not exist at: '.../envs/bearing-fault-diagnosis'
```
it means Section 4 never completed successfully, or you're on a
mismatched root-prefix. Run `mamba env list` to see what actually exists and
where, then re-run Section 4 if the environment is missing.

**If you see an error about the shell not being initialized**, go back to
Section 2 — you skipped or need to redo the shell hook.

---

## 6. Register the Jupyter kernel

Run this once, inside the activated environment, on **either OS**:

```bash
python -m ipykernel install --user --name bearing-fault-diagnosis --display-name "Bearing Fault Diagnosis"
```

This makes the environment selectable as a kernel inside Jupyter Lab/Notebook
and in VS Code's notebook interface.

---

## 7. Verify the install

With the environment activated, run:

```bash
python -c "import scipy.io, numpy, sklearn, tensorflow, torch; print('All core imports OK')"
```

If this prints without errors, you're set up correctly. If you get a version
conflict error (e.g. a numpy mismatch), see Section 9.

To start working:
```bash
jupyter lab
```
and select the "Bearing Fault Diagnosis" kernel from the kernel picker.

---

## 8. Adding a new package later

Don't `pip install` or `conda install` something ad hoc and move on — it
won't exist for anyone else on the team. Instead:

1. Add the package to `environment-full.yml` (conda-installable packages
   under `dependencies:`, pip-only packages under the `pip:` block).
2. Commit and push the change.
3. Everyone else pulls the change and runs, from the repo root:
   ```bash
   mamba env update -f environment-full.yml --prune
   ```

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Shell not initialized` | Section 2 was skipped | Run the `mamba shell init` command for your OS and shell, then reopen the terminal |
| `Cannot activate, prefix does not exist` | Environment was never created, or wrong name typed | Run `mamba env list` to check, then `mamba env create -f environment-full.yml` if missing |
| `X requires numpy>=A, but you have numpy=B` | Two packages (commonly TensorFlow and a newer library) pin conflicting numpy ranges | Check which package's pin is outdated, bump its version in `environment-full.yml`, and re-run `mamba env update --prune` |
| PowerShell: "running scripts is disabled" | Default Windows execution policy blocks conda/mamba's activation script | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Notebook can't find a package that's clearly installed | Wrong kernel selected | Re-run Section 6, then pick "Bearing Fault Diagnosis" from the kernel picker in Jupyter |
| File paths work on Mac but break on Windows for the same teammate | Hardcoded `/` paths or backslashes in code | Use `pathlib.Path` in Python code instead of manual path strings |

If none of these match what you're seeing, paste the exact error text in the
team channel before trying random fixes — most environment errors are easier
to diagnose from the literal message than from a guess.