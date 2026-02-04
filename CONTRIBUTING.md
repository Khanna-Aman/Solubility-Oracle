# 🤝 Contributing to Solubility Oracle

Thank you for your interest in contributing to Solubility Oracle! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all backgrounds and experience levels.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Solubility-Oracle.git
   cd Solubility-Oracle
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Khanna-Aman/Solubility-Oracle.git
   ```

## Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Intel GPU (optional, for acceleration)

### Python Environment
```bash
python -m venv .venv-solubility
.venv-solubility\Scripts\activate  # Windows
# source .venv-solubility/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Frontend Environment
```bash
cd frontend
npm install
```

## Making Changes

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `model/description` - Model improvements

### Commit Messages
Follow conventional commits:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `model`

Example:
```
feat(models): add attention visualization to AttentiveFP

- Implement attention weight extraction
- Add visualization utilities
- Update API to return attention scores
```

## Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make changes** and commit

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature
   ```

5. **Open a Pull Request** on GitHub

### PR Checklist
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions
- [ ] PR description explains changes
- [ ] Model changes include performance metrics

## Coding Standards

### Python
- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and testable

### JavaScript/TypeScript
- Use ESLint configuration
- Prefer functional components in React
- Use TypeScript for type safety

### Machine Learning
- Document model architecture changes
- Include training metrics and comparisons
- Validate on test set before PR
- Use reproducible random seeds

### General
- Keep functions small and focused
- Write meaningful variable names
- Add comments for complex logic
- Write tests for new features

## Testing

Run tests before submitting:
```bash
# Python tests
pytest tests/

# Frontend tests
cd frontend
npm test
```

## Questions?

Open an issue or start a discussion on GitHub!

---

**Thank you for contributing to Solubility Oracle!** 💧

