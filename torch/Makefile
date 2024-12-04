MODEL_ID="s3://controlplane-prod-uploads/densenet161.mar"
DOCKER_IMAGE="ghcr.io/beamlit/beamlit-runtime/torch:latest"
PLATFORM="linux/amd64"

build:
	docker build -t ${DOCKER_IMAGE} .

push:
	docker build --push --platform ${PLATFORM} -t ${DOCKER_IMAGE} .

run:
	docker rm -f torchserve
	docker run \
		--rm \
		--name torchserve \
		-p 80:80 \
		-p 8081:8081 \
		-e BL_CLIENT_CREDENTIALS=${CLIENT_CREDENTIALS} \
		${DOCKER_IMAGE} \
		--model-id ${BASE_URL}/${MODEL_ID}

test:
	curl http://127.0.0.1/predictions/densenet161 -T samples/shadow.jpg
	curl http://127.0.0.1/predictions/densenet161 -T samples/pixel.jpg

test-prod:
	curl --request POST \
		--url 'https://run.beamlit.dev/charlou/models/shadow/predictions/densenet161' \
		-T samples/shadow.jpg \
		--header 'X-Beamlit-Api-Key: ${API_KEY}'
