MODEL_ID=google-t5/t5-base
DOCKER_IMAGE=sindar/toto12345
TAG=latest

call:
	curl -X POST \
		http://localhost:80/ \
		-H "Content-Type: application/json" \
		-d '{"inputs": "My name is Sarah and I live in London"}'

build:
	docker build -t ${DOCKER_IMAGE}:${TAG} --platform linux/amd64 .

push: build
	docker push ${DOCKER_IMAGE}:${TAG}

run:
	docker rm -f blruntime
	docker run \
		--rm \
		--platform linux/amd64 \
		--name blruntime \
		-p 80:80 \
		${DOCKER_IMAGE}:${TAG} \
		--model-id $(MODEL_ID)