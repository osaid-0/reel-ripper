# Contributing

Bug reports and focused pull requests are welcome.

Before opening an issue:

1. Upgrade with `python -m pip install -U -r requirements.txt`.
2. Run `python main.py --help` and the relevant command with one public test URL.
3. For CUDA problems, include the output of `python main.py gpu-check`.
4. Run `python -m unittest discover -s tests -v`.

Include your operating system, Python version, command, and complete error text.
Never post browser cookies, `HF_TOKEN`, private reel URLs, or files from `out/`
unless you have reviewed and intentionally sanitized them.

Keep `main.py` as the single executable application file. Tests and project
documentation may live in their normal folders.
