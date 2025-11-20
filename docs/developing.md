
# Developing emlearn

For those that wish to hack on emlearn-micropython itself.

Contribution guidelines can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

If you just wish to use it as a library, see instead the [user guide](https://emlearn.readthedocs.io/en/latest/user_guide.html).

#### Prerequisites

These instructions have been tested on Linux.
They might also work on Mac OS.
For Windows, I recommend using Windows Subsystem for Linux (WSL2).

You will need to have **Python 3.10+ or later** already installed.


#### Download the code

Clone the repository using git
```
git clone https://github.com/emlearn/emlearn
```


#### Download dependencies

Fetch git submodules

```
git submodule update --init
```

NOTE: Recommend using a Python virtual environment (using `venv`, `uv`, etc.)

Install Python packages
```
pip install -r requirements.txt -r requirements.dev.txt
```

## Automated tests

#### Run tests on PC

Tests are found in the `./test` subdirectory.

```
python -m pytest -v ./test/
```

This will run both Python tests, as well as compile C tests and run them as a git submodule.



## Building documentation

Make sure to have dev dependencies installed already.

Run the Sphinx build

```
make -C docs/ html
```

Open the frontpage in browser `docs/_build/html/index.html`

