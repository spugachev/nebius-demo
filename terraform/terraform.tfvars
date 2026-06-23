# Nebius CSA assignment — 2-node H200 Soperator cluster
# Tenant: csa-hiring-sandboxK  Project: demoday-sergey

company_name = "spugachev"

production            = false
iam_merge_request_url = ""

#----------------------------------------------------------------------------------------------------------------------#
#                                                    Infrastructure                                                    #
#----------------------------------------------------------------------------------------------------------------------#

region         = "eu-north2"
iam_project_id = "project-e02qk7z0pr005jndcar132"
iam_tenant_id  = "tenant-e00znpks4hqd7c0e72"
vpc_subnet_id  = "vpcsubnet-e02g2fe36gdvbzke5b"

# Same tenant for observability; profile empty because public_o11y_enabled = false
o11y_iam_tenant_id = "tenant-e00znpks4hqd7c0e72"
o11y_profile       = ""

#----------------------------------------------------------------------------------------------------------------------#
#                                                        Storage                                                       #
#----------------------------------------------------------------------------------------------------------------------#

controller_state_on_filestore = false

filestore_controller_spool = {
  spec = {
    size_gibibytes       = 128
    block_size_kibibytes = 4
    forbid_deletion      = false
  }
}

# Shared jail filesystem: model weights (70GB) + dataset + 2 checkpoints (140GB) = ~300GB. 1TB gives headroom.
filestore_jail = {
  spec = {
    size_gibibytes       = 1024
    block_size_kibibytes = 4
    forbid_deletion      = false
  }
}

# /data submount for model weights, dataset, and checkpoints (~300GB needed, 500GiB gives headroom)
filestore_jail_submounts = [{
  name       = "data"
  mount_path = "/data"
  spec = {
    size_gibibytes       = 500
    block_size_kibibytes = 4
    forbid_deletion      = false
  }
}]

filestore_accounting = {
  spec = {
    size_gibibytes       = 128
    block_size_kibibytes = 4
    forbid_deletion      = false
  }
}

#----------------------------------------------------------------------------------------------------------------------#
#                                                      NFS in k8s                                                      #
#----------------------------------------------------------------------------------------------------------------------#

nfs_in_k8s = {
  enabled         = true
  version         = "1.2.0"
  use_stable_repo = true
  size_gibibytes  = 558
  disk_type       = "NETWORK_SSD_IO_M3"
  filesystem_type = "ext4"
  threads         = 64
}

#----------------------------------------------------------------------------------------------------------------------#
#                                                         Slurm                                                        #
#----------------------------------------------------------------------------------------------------------------------#

slurm_operator_version = "4.1.0"
slurm_operator_stable  = true

slurm_nodesets_partitions = [
  {
    name               = "main"
    is_all             = true
    slurm_nodeset_refs = []
    config             = "Default=YES PriorityTier=10 PreemptMode=OFF MaxTime=INFINITE State=UP OverSubscribe=YES"
  },
  {
    name               = "hidden"
    is_all             = true
    slurm_nodeset_refs = []
    config             = "Default=NO PriorityTier=10 PreemptMode=OFF Hidden=YES MaxTime=INFINITE State=UP OverSubscribe=YES"
  },
]

slurm_partition_config_type = "default"

#----------------------------------------------------------------------------------------------------------------------#
#                                                         Nodes                                                        #
#----------------------------------------------------------------------------------------------------------------------#

slurm_nodeset_system = {
  min_size = 3
  max_size = 9
  resource = {
    platform = "cpu-d3"
    preset   = "16vcpu-64gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD"
    size_gibibytes       = 128
    block_size_kibibytes = 4
  }
}

slurm_nodeset_controller = {
  size = 1
  resource = {
    platform = "cpu-d3"
    preset   = "16vcpu-64gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD"
    size_gibibytes       = 128
    block_size_kibibytes = 4
  }
}

# 2 x H200 SXM (8 GPUs each = 16 GPUs total)
# infiniband_fabric = "" → Nebius auto-assigns both nodes to the same IB fabric
slurm_nodeset_workers = [
  {
    name = "worker"
    size = 2
    autoscaling = {
      enabled  = false
      min_size = null
    }
    resource = {
      platform = "gpu-h200-sxm"
      preset   = "8gpu-128vcpu-1600gb"
    }
    boot_disk = {
      type                 = "NETWORK_SSD"
      size_gibibytes       = 512
      block_size_kibibytes = 4
    }
    gpu_cluster = {
      infiniband_fabric = "eu-north2-a"
    }
    preemptible      = null
    features         = null
    create_partition = null
    ephemeral_nodes                = false
    initial_number_ephemeral_nodes = 1
    persistent_volume_claim_retention_policy = {
      when_deleted = "Delete"
      when_scaled  = "Delete"
    }
    max_pods                  = 32
    node_local_jail_submounts = []
    node_local_image_disk = {
      enabled = false
    }
  },
]

use_preinstalled_gpu_drivers = true

slurm_nodeset_login = {
  size = 1
  resource = {
    platform = "cpu-d3"
    preset   = "16vcpu-64gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD"
    size_gibibytes       = 256
    block_size_kibibytes = 4
  }
}

slurm_nodeset_accounting = {
  resource = {
    platform = "cpu-d3"
    preset   = "16vcpu-64gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD"
    size_gibibytes       = 128
    block_size_kibibytes = 4
  }
}

slurm_nodeset_nfs = {
  size = 1
  resource = {
    platform = "cpu-d3"
    preset   = "32vcpu-128gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD"
    size_gibibytes       = 128
    block_size_kibibytes = 4
  }
}

#----------------------------------------------------------------------------------------------------------------------#
#                                                         Login                                                        #
#----------------------------------------------------------------------------------------------------------------------#

slurm_login_public_ip           = true
tailscale_enabled               = false
slurm_sssd_enabled              = false
slurm_sssd_conf_secret_ref_name = ""
slurm_sssd_ldap_ca_config_map_ref_name = ""

slurm_login_ssh_root_public_keys = [
  "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBx9GTsQyOngl1GO4WmnfJLakPKP2AnfKtYtRXJ7kI+0 nebius-demo-soperator",
]

#----------------------------------------------------------------------------------------------------------------------#
#                                                       Telemetry                                                      #
#----------------------------------------------------------------------------------------------------------------------#

slurm_exporter_enabled   = true
telemetry_enabled        = true
dcgm_job_mapping_enabled = true

soperator_notifier = {
  enabled = false
}

# Known Terraform recipe bug — must be false
public_o11y_enabled = false

#----------------------------------------------------------------------------------------------------------------------#
#                                                       Accounting                                                     #
#----------------------------------------------------------------------------------------------------------------------#

accounting_enabled = true

#----------------------------------------------------------------------------------------------------------------------#
#                                                        Backups                                                       #
#----------------------------------------------------------------------------------------------------------------------#

backups_enabled        = "auto"
backups_password       = "nebius-demo-backup-2026"
backups_schedule       = "@daily-random"
backups_prune_schedule = "@daily-random"

backups_retention = {
  keepDaily = 3
}

cleanup_bucket_on_destroy = false

#----------------------------------------------------------------------------------------------------------------------#
#                                                      Kubernetes                                                      #
#----------------------------------------------------------------------------------------------------------------------#

k8s_version = 1.33

nvidia_config_lines = [
  "options nvidia NVreg_RestrictProfilingToAdminUsers=0",
  "options nvidia NVreg_EnableStreamMemOPs=1",
  "options nvidia NVreg_RegistryDwords=\"PeerMappingOverride=1;\"",
]

active_checks_scope = "essential"

slurm_shared_memory_size_gibibytes = 64
maintenance_ignore_node_groups     = ["controller", "nfs"]
