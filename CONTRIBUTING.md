# Contributing to Digital Collections Explorer

Thank you for your interest in contributing to Digital Collections Explorer! This document provides guidelines and information to help you contribute effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Research Contributions](#research-contributions)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include as many details as possible using our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md):

- Clear and descriptive title
- Steps to reproduce the behavior
- Expected vs. actual behavior
- Screenshots or error logs
- Environment details (OS, Python/Node versions, collection type)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide a detailed description of the proposed functionality
- Explain why this enhancement would be useful
- Include mockups or examples if applicable

### Research Contributions

As an academic project, we especially welcome:

- **New CLIP model fine-tuning approaches** for digital collections
- **Evaluation methodologies** and benchmark datasets
- **User studies** or usability research findings
- **Domain-specific optimizations** for different collection types
- **Performance improvements** in embedding generation or search

Please open an issue or discussion to share your research ideas and findings.

### Documentation Improvements

- Fix typos or clarify existing documentation
- Add examples or tutorials
- Translate documentation (if multilingual support is needed)
- Improve API documentation or code comments

### Code Contributions

We welcome code contributions including:

- Bug fixes
- New features
- Performance improvements
- Code refactoring
- Test coverage improvements

## Development Setup

### Backend Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/digital-collections-explorer.git
   cd digital-collections-explorer
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure the project**
   ```bash
   npm install
   npm run setup -- --with=3d,pdf,zoom
   pip install -r requirements-3d.txt -r requirements-pdf.txt
   ```

4. **Index some test images** (optional, for testing)
   ```bash
   # Add some test images and PDFs to data/raw
   python -m src.ingest data/raw --data-dir data/collections/dev
   ```

5. **Start the backend server**
   ```bash
   DCE_DATA_DIR=data/collections/dev python -m src.backend.main
   ```

### Frontend Setup

1. **Navigate to the frontend directory**
   ```bash
   cd src/frontend/default
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

The frontend will be available at http://localhost:5173 and will proxy API requests to http://localhost:8000.

Each optional viewer is a package of its own in `src/frontend/default/viewers/`, with its own `package.json`. Setup installs a viewer only when `media` in `config.json` lists it, and the build leaves out the others. A new viewer exports its components from `index.js`, may add to the Vite configuration in `vite.js`, and is reached through `viewer('<name>')` in `src/services/viewers.js`.

### Making a frontend

A frontend that looks or works differently can replace the default one, and is best kept in a repository of its own. It needs only to:

1. Build with `npm install && npm run build` into a `dist/` folder holding an `index.html`.
2. Refer to its files by relative paths, as Vite does with `base: './'`, so that the site also works under a path of another website.
3. Talk to the backend only through its HTTP API: `/api/...` (listed at `/docs` on a running site), `/images/{id}?size=thumbnail` or `full`, `/models/{id}` for 3D models, and `/documents/{id}` for PDFs.

The backend serves `dist/` at the root of the site and fills the `<title>` and `<meta name="description">` of `index.html` from `collection.json`. A frontend may also write `dist/frontend.json`, as `{"media": ["3d"]}`, to say which kinds of material it shows; `python -m src.environment` reports it.

To use one, clone it into `src/frontend/<name>` and run `npm run setup -- --frontend=<name>`, or give `--frontend` the path to a folder elsewhere.

## Coding Standards

### Python

- **Formatting**: Use `black` with default settings
  ```bash
  black .
  ```

- **Import sorting**: Use `isort` with black-compatible profile
  ```bash
  isort .
  ```

- **Style guidelines**:
  - Follow PEP 8
  - Use type hints where applicable
  - Write docstrings for public functions and classes
  - Keep functions focused and single-purpose

- **File organization**:
  - `api/routes/`: HTTP endpoint handlers
  - `services/`: Business logic and ML operations
  - `models/schemas.py`: Pydantic models for validation
  - `core/`: Configuration and shared utilities

### JavaScript/React

- **Linting**: Run ESLint before committing
  ```bash
  npm run lint
  ```

- **Style guidelines**:
  - Use functional components with hooks
  - Keep components small and reusable
  - Use meaningful variable and function names
  - Add comments for complex logic

- **Component structure**:
  - One component per file
  - Co-locate CSS with components
  - Use absolute imports where configured

## Commit Guidelines

Write the subject line as a short sentence that says what the change does for the people who use or run the site, the way the history reads, for example "Cache images and assets under the base path too" or "Take the model type from the model's name when config.json gives only the name". Keep each commit to one change, so it can be reviewed and reverted on its own.

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes**
   - Write clear, focused commits
   - Add tests if applicable
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   # Python
   black .
   isort .
   pytest  # if you added tests
   
   # JavaScript (in frontend directory)
   npm run lint
   ```

4. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Use the PR template
   - Reference related issues
   - Provide clear description of changes
   - Add screenshots for UI changes

6. **Code Review**
   - Address review feedback
   - Keep PR focused on a single concern
   - Be responsive to comments

7. **Merge**
   - PRs require approval from maintainers
   - Ensure all CI checks pass
   - Squash commits may be requested for cleaner history

## Testing Guidelines

### Python Tests

We use `pytest` for Python testing:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_embedding_service.py

# Run with coverage
pytest --cov=src
```

**Test guidelines:**
- Write unit tests for `services/` logic
- Add integration tests for API routes
- Test edge cases and error handling
- Mock external dependencies (CLIP model, file I/O)

### Frontend Tests

Currently, frontend testing is minimal. Contributions to improve test coverage are welcome:

- Unit tests for utility functions
- Component tests with React Testing Library
- Integration tests for search workflows

## Documentation

### Code Documentation

- **Python**: Use docstrings (Google or NumPy style)
  ```python
  def generate_embedding(image_path: str) -> torch.Tensor:
      """Generate CLIP embedding for an image.
      
      Args:
          image_path: Path to the input image file
          
      Returns:
          torch.Tensor: The embedding vector
          
      Raises:
          FileNotFoundError: If image file doesn't exist
      """
  ```

- **JavaScript**: Use JSDoc for complex functions
  ```javascript
  /**
   * Performs semantic search using text or image input
   * @param {string} query - Text query or image URL
   * @param {number} limit - Maximum number of results
   * @returns {Promise<Array>} Search results
   */
  ```

### README and Guides

- Keep README.md up to date with new features
- Add examples for new functionality
- Update configuration documentation
- Create tutorial content for complex workflows

## Research Contributions

### Publishing Results

If you use this project in research:

1. **Cite the project**: Use the DOI badge in README.md
2. **Share your findings**: Open an issue or PR with your results
3. **Contribute improvements**: If your research leads to better models or methods, consider contributing them back

### Datasets and Models

- **Datasets**: If you create benchmark datasets, consider sharing them (with appropriate licenses)
- **Fine-tuned models**: Share model weights via Hugging Face or similar platforms
- **Evaluation scripts**: Contribute evaluation code to help others reproduce results

### Academic Collaboration

We welcome collaboration with researchers:

- Joint development of new features
- Evaluation and benchmarking studies
- User research and usability studies
- Domain-specific applications

Please reach out via issues or email to discuss research collaborations.

## Getting Help

- **Documentation**: Check README.md and code comments
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: For private inquiries or collaboration proposals

## Recognition

Contributors will be:

- Listed in project documentation
- Acknowledged in academic papers (if applicable)
- Given credit in release notes

Thank you for contributing to Digital Collections Explorer! 🎉
