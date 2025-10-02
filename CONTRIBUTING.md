# Contributing to Speeksee

🎉 Thank you for your interest in contributing to Speeksee!  
This document provides guidelines for contributing to the project.

---

## 1. Creating Issues

- Use **GitHub Issues** for bug reports, feature requests, or documentation improvements.
- When opening an issue, please include:
  - A clear description of the problem or suggestion
  - Steps to reproduce (if applicable)
  - Expected vs. actual behavior
  - Screenshots or logs, if available

---

## 2. Branching Strategy

- `main` branch → **protected** (no direct pushes)
- Branch naming convention: `<type>/#<issue-number>`
- Create feature branches for new work:

```bash
git checkout -b feat/#1
```

- For bug fixes:

```bash
git checkout -b bug/#2
```

- For refactoring:

```bash
git checkout -b refactor/#3
```

- Other types: `docs/#4`, `test/#5`, `chore/#6`, etc.

---

## 3. Pull Request (PR) Process

1. Fork the repository (if you're an external contributor) or create a new branch (if you're part of the organization).

2. Implement your changes and commit them.

```bash
git commit -m "feat: commit message"
git push origin feat/#1
```

3. Open a Pull Request:
   - Base branch: `main`
   - Compare branch: your feature/fix branch
   - Provide a clear title and description for your PR:
     - What problem it solves
     - Key changes made
     - **Include closing keywords in the PR description to automatically close related issues:**
       - `Closes #1` or `Fixes #1` or `Resolves #1`
       - Example: "Closes #1" will automatically close issue #1 when the PR is merged

4. Respond to reviewer feedback and make necessary updates.

5. **Review and Approval Process:**
   - All PRs must be reviewed by at least one maintainer before merging
   - Maintainers will review your code for:
     - Code quality and adherence to project standards
     - Potential bugs or issues
     - Alignment with project goals
   - Once approved by a maintainer, your PR will be merged into `main`
   - This project follows the MIT License - all contributions will be under the same license

---

## 4. Code Style & Commit Conventions

- Follow existing code conventions in the repository.
- Use automated formatters/linting tools where available.
- Commit messages should follow [Conventional Commits](https://www.conventionalcommits.org/)

---

## 5. Communication

- For significant changes, please discuss them in an issue before submitting a PR.
- Keep pull requests focused and manageable.
- Be open to feedback during the review process 🙂

---

🙌 Your contributions help make Speeksee better for everyone.  
Thank you for supporting open source!
