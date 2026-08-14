output "ecr_repository_url" {
  value = aws_ecr_repository.churn_api.repository_url
}

output "api_load_balancer_dns" {
  value = aws_lb.api.dns_name
}

output "mlflow_artifact_bucket" {
  value = aws_s3_bucket.mlflow_artifacts.bucket
}
