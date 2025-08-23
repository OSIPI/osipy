# Guide to Run Documentation Locally

## 1. Install Poetry

Run the following command in your project directory:

```bash
pip install poetry
```

This will install [Poetry](https://python-poetry.org/).

---

## 2. Install Dependencies

* **On macOS/Linux:**

```bash
poetry install
```

* **On Windows (Git Bash / PowerShell):**

```bash
poetry install
```

---

## 3. Activate the Virtual Environment

Find the environment path and activate it:

```bash
poetry env activate
```

This will print the full path of the environment. Copy it and source it manually, for example:

```bash
source /path/to/venv/bin/activate
```

Once activated, you’re inside the Poetry-managed virtual environment.

---

## 4. Serve MkDocs Locally

Run the following command to start the documentation server:

```bash
mkdocs serve
```

Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000) to view the docs.

---

## 5. (Optional) Deactivate the Virtual Environment

When you’re done working:

```bash
deactivate
```