#!/bin/bash
# deploy.sh — Build and deploy the prediction market bot via AWS SAM.
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - AWS SAM CLI installed (brew install aws-sam-cli / pip install aws-sam-cli)
#   - KALSHI_SECRET_ARN environment variable set (Secrets Manager ARN)
#   - SAM_BUCKET env var set (or default is used — must exist in your account)
#
# Usage:
#   export KALSHI_SECRET_ARN="arn:aws:secretsmanager:us-east-1:123456789:secret:kalshi-creds"
#   bash infra/deploy.sh

set -euo pipefail

STACK_NAME="${STACK_NAME:-prediction-market-bot}"
REGION="${AWS_REGION:-us-east-1}"
SAM_BUCKET="${SAM_BUCKET:-prediction-market-bot-deploy}"

# Validate required env vars
if [ -z "${KALSHI_SECRET_ARN:-}" ]; then
    echo "ERROR: KALSHI_SECRET_ARN environment variable is required."
    echo "  export KALSHI_SECRET_ARN=\"arn:aws:secretsmanager:REGION:ACCOUNT:secret:NAME\""
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# 1. Prepare the dependencies layer
# ---------------------------------------------------------------------------
LAYER_DIR="$PROJECT_ROOT/layers/dependencies/python"
echo "==> Preparing dependencies layer..."
rm -rf "$PROJECT_ROOT/layers/dependencies"
mkdir -p "$LAYER_DIR"

# Install production deps (exclude dev/test packages)
pip install \
    --target "$LAYER_DIR" \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --implementation cp \
    --python-version 3.11 \
    --upgrade \
    requests scipy numpy pandas pyarrow boto3 python-dotenv cryptography 2>&1 | tail -5

echo "    Layer size: $(du -sh "$PROJECT_ROOT/layers/dependencies" | cut -f1)"

# ---------------------------------------------------------------------------
# 2. SAM build
# ---------------------------------------------------------------------------
echo "==> Building SAM application..."
cd "$SCRIPT_DIR"
sam build --template-file template.yaml

# ---------------------------------------------------------------------------
# 3. SAM deploy
# ---------------------------------------------------------------------------
echo "==> Deploying stack: $STACK_NAME to $REGION..."
sam deploy \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --s3-bucket "$SAM_BUCKET" \
    --capabilities CAPABILITY_IAM \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset \
    --parameter-overrides \
        "KalshiSecretArn=$KALSHI_SECRET_ARN"

# ---------------------------------------------------------------------------
# 4. Print outputs
# ---------------------------------------------------------------------------
echo ""
echo "==> Deployment complete."
aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs" \
    --output table 2>/dev/null || true

echo ""
echo "Stack: $STACK_NAME | Region: $REGION"
echo "To invoke manually:"
echo "  aws lambda invoke --function-name ${STACK_NAME}-bot --payload '{\"action\":\"morning_scan\"}' /dev/stdout"
