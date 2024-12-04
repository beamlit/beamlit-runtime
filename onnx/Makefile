DOCKER_IMAGE=kibaes/onnxruntime-server:1.19.2-linux-cpu

run:
	docker rm -f onnx-server || true
	docker run --name onnx-server -ti --rm \
		-p 80:80 \
		--platform linux/amd64 \
		-v "/Users/cdrappier/Workspace/beamlit-runtime/onnx:/app/models" \
		-e "ONNX_SERVER_SWAGGER_URL_PATH=/api-docs" \
		${DOCKER_IMAGE}

call:
	curl -X POST \
		http://localhost:8001/v1/models/model:predict \
		-H "Content-Type: application/json" \
		-d '{"input_ids": [[101, 2054, 2003, 102, 0, 0, 0, 0, 0, 0]],"attention_mask": [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]}'
