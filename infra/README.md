# AWS Infrastructure (Terraform)

Reference infrastructure-as-code for deploying the churn API to AWS ECS Fargate.

## What it creates
- **ECR** repository for the API Docker image
- **S3** bucket for the MLflow artifact store (versioned + encrypted)
- **ECS Fargate cluster + service** running the API behind an **ALB**
- **IAM** task execution role scoped to S3 read + ECS execution
- **CloudWatch** log group for API logs

## Usage

```bash
cd infra
terraform init
terraform apply \
  -var="vpc_id=vpc-xxxxxxxx" \
  -var='subnet_ids=["subnet-aaaa","subnet-bbbb"]'
```

After apply, push the image the GitHub Actions `cd.yml` workflow builds:

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -f docker/Dockerfile.api -t <ecr_repository_url>:latest .
docker push <ecr_repository_url>:latest
aws ecs update-service --cluster churn-mlops-cluster --service churn-mlops-api --force-new-deployment
```

## Not included (kept out of scope for the demo, noted for realism)
- RDS Postgres backend store for MLflow (SQLite is used in the Docker demo; swap
  `docker/Dockerfile.mlflow` command for a Postgres URI in real prod)
- VPC/subnet provisioning (assumes an existing VPC, passed as variables)
- Autoscaling policies, WAF, HTTPS/ACM certificate on the ALB
- Secrets Manager for credentials (env vars are used in the demo)

This project is designed to run **fully locally via Docker Compose** for the
live demo; the Terraform here demonstrates how it would be productionized on
AWS without requiring an actual AWS account to try the demo.
