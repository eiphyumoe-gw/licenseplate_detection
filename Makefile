docker-build:
	sudo docker build -t yolo_docker -f docker/Dockerfile .

docker-run:
	sudo docker run -it --rm -v ./:/Test-docker/ --gpus all --network=host yolo_docker