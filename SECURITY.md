# Security Policy

PiMRef is a research framework for **detecting** spear-phishing emails. This
policy covers vulnerabilities in *this codebase* (e.g. unsafe parsing of
untrusted email content, rendering, or the web/expansion agent), **not** phishing
emails themselves.

## Supported Versions

This is research software released without long-term support guarantees. Security
fixes are applied to the `main` branch only. Please always run the latest commit.

| Version | Supported          |
| ------- | ------------------ |
| `main`  | :white_check_mark: |
| older   | :x:                |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report privately:

- Preferred: open a [GitHub Security Advisory](https://github.com/your-org/PhishEmail/security/advisories/new)
  (Security → Report a vulnerability), **or**
- Email: **`TODO: add a security contact email before publishing`**

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce (a minimal proof-of-concept, e.g. a crafted `.eml`, is ideal).
- The affected file(s)/component and the commit hash you tested.

### What to expect

- We aim to acknowledge reports within a reasonable timeframe.
- We will keep you informed of remediation progress and coordinate a disclosure
  timeline with you.
- We will credit reporters in the release notes unless you prefer to remain
  anonymous.

## Handling Untrusted Input

This project parses, renders, and runs OCR/ML over **untrusted email content**.
When deploying or extending it:

- Run inference in a sandboxed / least-privilege environment.
- Treat all parsed addresses, URLs, headers, and rendered HTML as hostile input.
- The expansion agent makes outbound network/API calls; restrict egress as
  appropriate for your environment.
