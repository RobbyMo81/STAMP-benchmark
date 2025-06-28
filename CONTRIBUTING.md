## 🛠️ CONTRIBUTING to STAMP

First off—thanks for your interest in improving **STAMP**! Whether you're fixing bugs, adding benchmark features, or enhancing visualizations, every contribution helps us build a better tool for the ML performance community.

---

### 📋 Before You Begin

- Review our [LICENSE](./LICENSE) — STAMP is open for **non-commercial contributions only**.
- Check the [README](./README.md) for project goals, architecture, and current features.
- Browse [open issues](https://github.com/your-org/STAMP-benchmark/issues) to see where help is needed—or propose your own!

---

### 🧭 How to Contribute

#### 1. **Fork the Repo**
```bash
git clone https://github.com/your-username/STAMP-benchmark
cd STAMP-benchmark
```

#### 2. **Create a New Branch**
```bash
git checkout -b feat/your-feature-name
```

#### 3. **Make Your Changes**
- Follow existing naming and formatting conventions.
- Log performance results to `logs/` if applicable.
- If you're touching core benchmarking logic, write a test or add a note in the PR explaining your reasoning.

#### 4. **Submit a Pull Request**
- Provide a clear description of your changes.
- Link any related issue.
- Bonus: include screenshots or sample logs to illustrate your work.

---

### 💡 Contributor Tips

- 📦 Add new frameworks or operators in the `benchmarks/` folder.
- 📈 Enhance visual analytics via Streamlit in `scripts/benchmark_dashboard.py`.
- 🧪 Use `main.py` for CLI integrations and argument parsing.
- 🔐 If adding dependencies, update `requirements.txt` and justify in PR.

---

### 🏗️ Development Setup

#### Local Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Format code
black .
flake8 .

# Type checking
mypy .
```

#### Docker Development
```bash
# Build and run with Docker Compose
docker-compose up --build

# Run specific services
docker-compose up stamp-benchmark
docker-compose up dashboard
```

---

### 📝 Code Style Guidelines

- **Python**: Follow PEP 8 standards
- **Formatting**: Use `black` for code formatting
- **Linting**: Ensure `flake8` passes without errors
- **Type Hints**: Add type annotations for new functions
- **Docstrings**: Use Google-style docstrings for modules and functions

Example:
```python
def benchmark_operation(
    framework: str, 
    precision: str, 
    matrix_size: int
) -> Dict[str, Any]:
    """Benchmark a tensor operation across frameworks.
    
    Args:
        framework: Target ML framework ('pytorch', 'tensorflow', 'jax')
        precision: Precision mode ('fp16', 'fp32', 'fp64', 'mixed')
        matrix_size: Size of matrices for benchmarking
        
    Returns:
        Dictionary containing benchmark results and metadata
    """
    pass
```

---

### 🧪 Testing Guidelines

- Write unit tests for new functionality in `tests/`
- Include integration tests for end-to-end workflows
- Test across different precision modes and frameworks
- Verify GPU memory tracking accuracy

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_benchmarks.py
pytest tests/test_profiling.py

# Run with coverage
pytest --cov=. --cov-report=html
```

---

### 📊 Benchmark Implementation Guide

When adding new benchmarks:

1. **Create framework-specific implementations** in respective directories:
   - `benchmarks/pytorch/`
   - `benchmarks/tensorflow/`
   - `benchmarks/jax/`

2. **Follow the base benchmark interface**:
```python
class BaseBenchmark:
    def setup(self) -> None:
        """Initialize benchmark resources."""
        
    def run(self) -> Dict[str, Any]:
        """Execute benchmark and return results."""
        
    def cleanup(self) -> None:
        """Clean up resources after benchmark."""
```

3. **Include comprehensive logging**:
   - Execution time
   - Memory usage (GPU/CPU)
   - Precision-specific metrics
   - Error handling and recovery

---

### 🐛 Bug Reports

When reporting bugs, please include:

- **Environment details**: OS, Python version, GPU specs
- **Framework versions**: PyTorch, TensorFlow, JAX versions
- **Steps to reproduce**: Minimal code example
- **Expected vs actual behavior**
- **Log files**: Relevant benchmark logs from `logs/`

---

### ✨ Feature Requests

For new features, please provide:

- **Use case description**: Why is this feature needed?
- **Proposed implementation**: High-level approach
- **Framework impact**: Which ML frameworks are affected?
- **Performance considerations**: Expected overhead or benefits

---

### ⚠️ Restrictions Reminder

Per the Modified MIT License, please do **not**:
- Submit code intended for commercial integration or monetization.
- Obfuscate benchmarking logic to hide performance regressions.

If your use case borders on commercial or enterprise usage, reach out first.

---

### 🤝 Code of Conduct

This project supports a respectful, inclusive, and collaborative space. Harassment, discrimination, or disrespectful behavior will not be tolerated.

**Our Standards:**
- Be respectful and constructive in discussions
- Focus on technical merit and project improvement
- Help newcomers learn and contribute effectively
- Report inappropriate behavior to project maintainers

---

### 📞 Getting Help

- **GitHub Issues**: For bugs, feature requests, and general questions
- **Discussions**: For broader topics and community support
- **Pull Request Reviews**: For code-specific questions and feedback

---

### 🎯 Priority Areas

We're especially looking for contributions in:

1. **New Framework Support**: Adding support for emerging ML frameworks
2. **Precision Modes**: Enhanced mixed-precision benchmarking
3. **Visualization**: Improved Streamlit dashboard features
4. **Memory Profiling**: More detailed GPU memory tracking
5. **Performance Optimization**: Faster benchmark execution
6. **Documentation**: Better guides and examples

---

Thank you for contributing to STAMP! 🚀
