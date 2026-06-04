# Examples

Tiny, fully synthetic emails for trying out PiMRef. All addresses use the
reserved `example.com` / `.example` domains and contain no real content — they
exist only to demonstrate the detection pipeline.

| File | What it illustrates |
| --- | --- |
| `sample_phishing.eml`   | Claims to be **PayPal** but is sent from an unrelated domain (`account-verify.example`) and contains a call-to-action ("verify your account") with a link — the impostor + intent signal PiMRef flags. |
| `sample_legitimate.eml` | A benign notification from a sender whose domain matches the claimed identity. |

## Run

From the project root (after `pixi install`, `pixi run playwright install chromium`,
and downloading checkpoints via `pixi run get-model`):

```bash
pixi run inference --email_dir examples/ --output_csv examples_results.csv
```

Then inspect `examples_results.csv`. For the phishing sample you should see a
recognized identity (PayPal), one or more `required_actions`, and `our_pred` =
`True`; the legitimate sample should be predicted benign.

> These files are for testing/educational use only.
