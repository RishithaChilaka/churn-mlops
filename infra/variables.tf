variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "churn-mlops"
}

variable "environment" {
  default = "demo"
}

variable "vpc_id" {
  description = "Existing VPC to deploy into"
  type        = string
}

variable "subnet_ids" {
  description = "Public subnet IDs for the ALB + Fargate tasks"
  type        = list(string)
}

variable "desired_count" {
  default = 1
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking server URI reachable from ECS tasks"
  type        = string
  default     = "http://mlflow.internal:5000"
}
