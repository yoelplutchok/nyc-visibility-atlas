# Contributing to NYC Visibility Atlas

Thank you for your interest in contributing to the NYC Visibility Atlas project!

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in [GitHub Issues](../../issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (Python version, OS)

### Submitting Changes

1. **Fork the repository**

2. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Set up the development environment:**
   ```bash
   conda env create -f environment.yml
   conda activate visibility-atlas
   pip install -e .
   ```

4. **Make your changes** following our coding standards (see below)

5. **Run the tests:**
   ```bash
   make test
   ```

6. **Commit your changes:**
   ```bash
   git commit -m "Brief description of changes"
   ```

7. **Push and create a Pull Request**

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints where practical
- Maximum line length: 100 characters
- Use `black` for formatting: `black src/ scripts/ tests/`
- Use `ruff` for linting: `ruff check src/ scripts/`

### Documentation

- All functions should have docstrings (Google style)
- Update README.md if adding new features
- Add to CHANGELOG.md for significant changes

### Pipeline Scripts

When modifying or adding pipeline scripts:

1. Follow the naming convention: `XX_description.py`
2. Use the project's logging utilities (`visibility_atlas.logging_utils`)
3. Use atomic writes (`visibility_atlas.io_utils`)
4. Write metadata sidecars for outputs
5. Validate schemas on input/output
6. Add a "Completed Step" log entry to `nyc_visibility_atlas_project_context.md`

### Testing

- Add tests for new functionality
- Run the full test suite before submitting
- Smoke tests should pass: `make test-smoke`

## Project Structure

```
scripts/     # Pipeline scripts (numbered 00-17)
src/         # Core Python package
tests/       # Test suite
configs/     # YAML configuration files
data/        # Data directories (mostly gitignored)
reports/     # Generated reports and documentation
```

## Questions?

Open a GitHub issue or discussion for questions about the project.

