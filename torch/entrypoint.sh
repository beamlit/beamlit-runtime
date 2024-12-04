#!/bin/bash

# Initialize variables
MODEL_ID=$MODEL_ID
MODEL_NAME=$MODEL_NAME
MODEL_PATH=${MODEL_PATH:-/home/model-server}
AWS_REGION=$AWS_REGION
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
OAUTH_TOKEN_URL=${OAUTH_TOKEN_URL:-https://api.beamlit.dev/v0/oauth/token}

parse_arguments()
{
    # Parse arguments
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --model-id) MODEL_ID="$2"; shift 2 ;;
            --model-path) MODEL_PATH="$2"; shift 2 ;;
            --model-name) MODEL_NAME="$2"; shift 2 ;;
            --aws-region) AWS_REGION="$2"; shift 2 ;;
            --aws-access-key-id) AWS_ACCESS_KEY_ID="$2"; shift 2 ;;
            --aws-secret-access-key) AWS_SECRET_ACCESS_KEY="$2"; shift 2 ;;
            *) echo "Unknown parameter passed: $1"; exit 1 ;;
        esac
    done

    # Check if model_id argument is provided
    if [ -z "$MODEL_ID" ]; then
        echo "Usage: $0 --model-id <model_id> [--model-name <model_name>] [--aws-region <aws_region>] [--aws-access-key-id <aws_access_key_id>] [--aws-secret-access-key <aws_secret_access_key>]"
        exit 1
    fi

    # If model name is not provided, use the basename of the model ID
    MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_ID")}
}

# Determine MODEL_PROTOCOL based on MODEL_ID
determine_model_protocol() {
    if [[ "${MODEL_ID}" == https://* ]]; then
        MODEL_PROTOCOL="https"
    elif [[ "${MODEL_ID}" == http://* ]]; then
        MODEL_PROTOCOL="http"
    elif [[ "${MODEL_ID}" == s3://* ]]; then
        MODEL_PROTOCOL="s3"
    else
        echo "Unsupported MODEL_ID format. Please use a valid URL."
        exit 1
    fi
}

download_http()
{
    if [ ! -f "$MODEL_PATH/model-store/$MODEL_NAME" ]; then
        echo "Downloading model from $MODEL_ID to $MODEL_PATH/model-store/$MODEL_NAME"

        if [ ! -z "$BL_CLIENT_CREDENTIALS" ]; then
            echo "Getting OAuth token..."
            RESPONSE=$(curl -s -X POST \
                -H "Authorization: Basic $BL_CLIENT_CREDENTIALS" \
                -H "Content-Type: application/x-www-form-urlencoded" \
                "$OAUTH_TOKEN_URL" \
                -d "grant_type=client_credentials")
            ACCESS_TOKEN=$(echo $RESPONSE | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

            if [ -z "$ACCESS_TOKEN" ]; then
                echo "Failed to get access token"
                exit 1
            fi
        fi

        curl -L -o "$MODEL_PATH/model-store/$MODEL_NAME" "$MODEL_ID" \
            -H "X-Beamlit-Authorization: Bearer $ACCESS_TOKEN"
        du -hs "$MODEL_PATH/model-store/$MODEL_NAME"
        echo ""
    else
        echo "File $MODEL_PATH/model-store/$MODEL_NAME already exists. Skipping download."
    fi
}

download_s3()
{
    echo "Downloading model from $MODEL_ID to $MODEL_PATH/model-store/$MODEL_NAME"
    if ! AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY aws s3 cp "$MODEL_ID" "$MODEL_PATH/model-store/$MODEL_NAME" \
        --region "$AWS_REGION"; then
        echo "Failed to download model from S3. Exiting."
        exit 1
    fi
}

download_model()
{
    mkdir -p "$MODEL_PATH/model-store"
    determine_model_protocol
    if [ "$MODEL_PROTOCOL" == "https" ]; then
        download_http
    elif [ "$MODEL_PROTOCOL" == "http" ]; then
        download_http
    elif [ "$MODEL_PROTOCOL" == "s3" ]; then
        download_s3
    else
        echo "Unsupported protocol: $MODEL_PROTOCOL"
        exit 1
    fi
}

generate_config_properties()
{
    echo "Generating config.properties file"
    cat <<EOL > "$MODEL_PATH/config.properties"
inference_address=http://0.0.0.0:80
management_address=http://0.0.0.0:8081
model_store=$MODEL_PATH/model-store
load_models=${MODEL_NAME}
disable_token_authorization=true
EOL
}


function start_torchserve()
{
    echo "Starting TorchServe..."
    torchserve --model-store "$MODEL_PATH/model-store" --ts-config "$MODEL_PATH/config.properties"
    tail -f /dev/null
}

parse_arguments "$@"
determine_model_protocol
download_model
generate_config_properties
start_torchserve
