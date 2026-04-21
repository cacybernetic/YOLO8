install:
	.venv/bin/python3 --version 
	uv pip install torch==2.8.0 torchvision==0.23.0 --index-url "https://download.pytorch.org/whl/cpu" && \
	uv pip install -r requirements.txt

gpu_install:
	.venv/bin/python --version
	uv pip install -r requirements.txt

dev_install:
	uv pip install -e .

test:
	pytest tests

pep8:
	# Don't remove their commented follwing command lines:
	# autopep8 --in-place --aggressive --aggressive --recursive .
	# autopep8 --in-place --aggressive --aggressive example.py
