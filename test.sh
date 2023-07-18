export PYTHONPATH=$PYTHONPATH:./:./build/lib.linux-x86_64-3.8
echo $PYTHONPATH
python3 setup.py build
# To run a single test only
# python3 -m pytest -v --cov=emlearn --cov-report html --cov-report term-missing --cov-branch test/ -k test_window_function_hann -s
python3 -m pytest -v --cov=emlearn --cov-report html --cov-report term-missing --cov-branch test/
