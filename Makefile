
.PHONY: train-mnist

init:
	poetry install

train-mnist-cpu: init
	(cd UnconditionalDiffusion && \
		poetry run python main.py --train --dataset mnist --batch_size=128 --imsize=28 --device=1 --channels=1 --accelerator=cpu --max_epochs=10)