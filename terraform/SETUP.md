# Terraform setup

This is a thin configuration on top of the official Nebius Soperator recipe. The
`.tf` files here (`main.tf`, `variables.tf`, `terraform.tf`, `driver_presets.tf`,
`terraform.tfvars`) reference recipe modules under `./vendor/` (for example
`./vendor/soperator/modules/slurm`), so you must place the recipe there before
`terraform init`.

## 1. Fetch the recipe into ./vendor

The recipe is the public `nebius-solution-library`. From this `terraform/` directory:

```bash
git clone https://github.com/nebius/nebius-solution-library.git vendor
```

(If this directory is part of the original git repo, the same content is wired as a
submodule and `git submodule update --init terraform/vendor` works instead.)

## 2. Prerequisites

- `terraform`, the Nebius CLI (authenticated), and `yq` installed
  (`yq` is required by the recipe at apply time).

## 3. Review terraform.tfvars

- `public_o11y_enabled = false` is already set (works around a known recipe issue).
- Update the tenant/project to your own (it currently targets the shared
  `csa-hiring-sandboxK` tenant used for this assignment).
- Cluster shape is 2 worker nodes x 8 H200 on fabric `eu-north2-a`, with a shared
  jail filesystem and a `/data` submount.

## 4. Apply

```bash
terraform init
terraform plan  -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

After apply, `login.sh` is written with the login-node address; run `bash login.sh`
to SSH in. Do not point two different Soperator jails at the same shared filesystem.
