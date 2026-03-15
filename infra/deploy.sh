#!/bin/bash
set -e

STACK_NAME="kxbtc-trading-bot"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "Building SAM template..."
sam build --use-container

echo ""
echo "Deploying to AWS..."
sam deploy \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --guided

echo ""
echo "Deployment complete!"
echo "Stack name: $STACK_NAME"
echo "Region: $AWS_REGION"
