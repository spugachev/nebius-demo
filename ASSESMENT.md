# Nebius Home Assignment

Source: email from Alexei Konovalov, received June 19, 2026.

Assignment period: 1 week from receipt, so the expected deadline is June 26, 2026.

## Context

A potential customer is considering reserving GPU capacity in Nebius:

- 512 H100 GPUs
- Initial reservation duration: 6 months
- Current stage: PoC (Proof of Concept)

Customer profile:

- Small VC-funded startup with fewer than 20 employees
- Builds process automation products with AI agents
- PoC team is mostly ML engineers
- Team has limited cloud and infrastructure expertise
- They want to test Nebius with an end-to-end fine-tuning workflow for an open-source LLM that performs function calling

## Objective

Prepare an end-to-end example that supports the customer during the PoC.

The main deliverable is a multi-node LLM fine-tuning example based on Nebius Soperator, using the allocated PoC capacity efficiently.

During the demo day, present the example to the customer and explain the technical choices.

## Provided PoC Capacity

- 2 H200 nodes
- 8 GPU cards per node
- Fabric: `eu-north2-a`
- 1 TB SSD network disk
- 1 TB SSD shared filesystem

The example should utilize the provided capacity efficiently and use more than 80% of the GPUs.

GPU utilization can be verified in the Nebius console monitoring dashboards.

## Exercise 1: LLM Fine-Tuning

Status expectation: required for passing the assignment.

Deliverables:

- End-to-end multi-node fine-tuning example
- Code example
- Documentation for reproduction
- Documentation for monitoring
- Short demo-day presentation

Implementation choices are open:

- Scheduler choice is up to the implementer
- Fine-tuning framework choice is up to the implementer
- Storage type choice is up to the implementer
- Dataset choice can be anything
- Model choice can be anything

The example must be based on Soperator and should use the allocated PoC capacity efficiently.

## Exercise 2: Inference

Status expectation: extra point.

Deliverables:

- Run inference on the same Kubernetes cluster
- Serve the trained model
- Run the original untrained model
- Compare the results of the trained and original models

Only a code example is required for this exercise.

## Submission

When the assignment is complete and ready to submit:

- Reply to the original email thread
- Explain what was successfully achieved
- Explain what was challenging
- Attach the Terraform code used for the deployment

Important: do not destroy the lab environment. It should remain available for the next demo-day meeting.

## Guidelines

- When accepting the platform invite, join the existing tenant `csa-hiring-sandboxK`.
- Do not create a new tenant.
- Avoid using the same shared filesystem for two different jails.
- There is an open issue in the Terraform recipe. In the `.tfvars` file, set:

```hcl
public_o11y_enabled = false
```

- Install the `yq` library from the shell where `terraform apply` is run.

## Useful Links

- Nebius CSA Solution Library: https://github.com/nebius/nebius-solution-library
- Kubernetes training solution: https://github.com/nebius/nebius-solution-library/tree/main/k8s-training
- Slurm solution: https://github.com/nebius/nebius-solution-library/tree/main/soperator
- Nebius AI Cloud documentation: https://docs.nebius.com/
- Slurm operator for Kubernetes, Soperator: https://nebius.com/services/soperator

## Acceptance Notes

- Completing Exercise 1 is considered a pass.
- Completing Exercise 2 is an extra point.
- Exercise 1 requires a short presentation during the demo day.
- Exercise 2 requires only a code example.
