install:
	@pip install -U pip
	@pip install .

reinstall:
	@pip uninstall data_sequences -y
	@pip install -U pip
	@pip install .

remove:
	@pip uninstall data_sequences -y

test:
	@coverage run -m pytest tests
