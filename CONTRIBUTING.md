# Contributing to SLM Fine-tuning

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit code changes
- 🧪 Add tests
- 📊 Share benchmarks

## Getting Started

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/SLM-fine-tunning.git
cd SLM-fine-tunning
```

3. Create a branch:
```bash
git checkout -b feature/your-feature-name
```

4. Make your changes

5. Test your changes:
```bash
python -m pytest tests/
python src/prepare_data.py --input data/raw/sample.txt --output /tmp/test
```

6. Commit and push:
```bash
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

7. Open a Pull Request

## Code Style

### Python

Follow PEP 8 guidelines:
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use docstrings for functions and classes
- Type hints are encouraged

Example:
```python
def process_data(input_text: str, max_length: int = 512) -> Dict[str, Any]:
    """
    Process input text for training.
    
    Args:
        input_text: Raw input text
        max_length: Maximum sequence length
        
    Returns:
        Processed data dictionary
    """
    # Implementation
    pass
```

### Documentation

- Use Markdown for documentation
- Include code examples
- Add links to related resources
- Keep formatting consistent

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style
- [ ] Tests pass (if applicable)
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No unnecessary files included

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How the changes were tested

## Related Issues
Closes #123
```

## Reporting Issues

### Bug Reports

Include:
- Python version
- CUDA version (if applicable)
- GPU model
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches
- Impact on existing functionality

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Dev dependencies

# Run tests
pytest tests/

# Format code
black src/ tests/

# Lint
flake8 src/ tests/
```

## Areas Needing Contributions

### High Priority

- [ ] Add more data preprocessing methods
- [ ] Implement evaluation metrics
- [ ] Add multi-GPU training support
- [ ] Create deployment examples
- [ ] Improve error handling

### Medium Priority

- [ ] Add more model configurations (Phi-3, Mistral)
- [ ] Create web UI for inference
- [ ] Add data augmentation techniques
- [ ] Implement few-shot learning examples
- [ ] Add performance benchmarks

### Low Priority

- [ ] Add visualization tools
- [ ] Create tutorials
- [ ] Add experiment tracking (MLflow)
- [ ] Implement model compression techniques

## Code Review Process

1. Maintainers review all PRs
2. Feedback provided within 3-5 days
3. Address review comments
4. Approved PRs merged to main

## Community

- Be respectful and constructive
- Help others learn
- Share knowledge and experiences
- Follow the Code of Conduct

## Questions?

- Open a GitHub issue
- Check existing documentation
- Review closed issues/PRs

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

Thank you for contributing! 🎉
