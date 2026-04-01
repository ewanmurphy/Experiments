"""Tests for cluster account limits module."""

import pytest
from experiment.cluster.limits import (
    AccountLimits,
    QoSLimits,
    UserResourceLimits,
    parse_sacctmgr_user_output,
    parse_scontrol_partition_output,
    format_limits_card,
    format_limits_detailed,
    format_limits_compact,
)


class TestParseSacctmgrUserOutput:
    """Test parsing sacctmgr user output."""

    def test_parse_user_with_limits(self):
        """Test parsing user with resource limits."""
        output = """Name|Account|MaxCpusPerUser|MaxRunningCpus|MaxJobs|MaxSubmitJobs|MaxNodesPerJob|MaxWallDurationPerJob|GrpCpuLimit|GrpJobLimit|QOS
testuser|default|128|64|10|20|4|24:00:00|256|50|normal,high"""

        limits = parse_sacctmgr_user_output(output, "testuser")

        assert limits is not None
        assert limits.username == "testuser"
        assert limits.account == "default"
        assert limits.max_cpus_per_user == 128
        assert limits.max_cpus_running == 64
        assert limits.max_jobs == 10
        assert limits.max_jobs_submit == 20
        assert limits.max_node_per_job == 4
        assert limits.max_wall_duration == "24:00:00"
        assert limits.grp_cpu_limit == 256
        assert limits.grp_job_limit == 50
        assert "normal" in limits.qos_list
        assert "high" in limits.qos_list

    def test_parse_user_with_unlimited(self):
        """Test parsing user with unlimited values."""
        output = """Name|Account|MaxCpusPerUser|MaxRunningCpus|MaxJobs|MaxSubmitJobs|MaxNodesPerJob|MaxWallDurationPerJob|GrpCpuLimit|GrpJobLimit|QOS
adminuser|default|None|None|None|None|None|None|None|None|normal"""

        limits = parse_sacctmgr_user_output(output, "adminuser")

        assert limits is not None
        assert limits.max_cpus_per_user is None
        assert limits.max_jobs is None
        assert limits.max_wall_duration is None

    def test_parse_user_not_found(self):
        """Test parsing when user not found."""
        output = """Name|Account|MaxCpusPerUser|MaxRunningCpus|MaxJobs|MaxSubmitJobs|MaxNodesPerJob|MaxWallDurationPerJob|GrpCpuLimit|GrpJobLimit|QOS
otheruser|default|128|64|10|20|4|24:00:00|256|50|normal"""

        limits = parse_sacctmgr_user_output(output, "testuser")

        assert limits is None

    def test_parse_multiple_users(self):
        """Test parsing output with multiple users."""
        output = """Name|Account|MaxCpusPerUser|MaxRunningCpus|MaxJobs|MaxSubmitJobs|MaxNodesPerJob|MaxWallDurationPerJob|GrpCpuLimit|GrpJobLimit|QOS
user1|default|64|32|5|10|2|12:00:00|128|25|normal
user2|default|128|64|10|20|4|24:00:00|256|50|normal,high
user3|default|256|128|20|40|8|48:00:00|512|100|normal"""

        limits = parse_sacctmgr_user_output(output, "user2")

        assert limits is not None
        assert limits.username == "user2"
        assert limits.max_cpus_per_user == 128

    def test_parse_single_qos(self):
        """Test parsing user with single QoS."""
        output = """Name|Account|MaxCpusPerUser|MaxRunningCpus|MaxJobs|MaxSubmitJobs|MaxNodesPerJob|MaxWallDurationPerJob|GrpCpuLimit|GrpJobLimit|QOS
testuser|default|128|64|10|20|4|24:00:00|256|50|normal"""

        limits = parse_sacctmgr_user_output(output, "testuser")

        assert limits is not None
        assert len(limits.qos_list) == 1
        assert limits.qos_list[0] == "normal"


class TestParseScontrolPartitionOutput:
    """Test parsing scontrol partition output."""

    def test_parse_partition_limits(self):
        """Test parsing partition limits from scontrol."""
        output = """PartitionName=gpu
   MaxCpusPerNode=128
   MaxNodes=4
   MaxTime=48:00:00
   PartitionName=gpu"""

        partition = parse_scontrol_partition_output(output)

        assert partition is not None
        assert partition.name == "gpu"
        assert partition.max_cpus_per_node == 128
        assert partition.max_nodes == 4
        assert partition.max_time == "48:00:00"

    def test_parse_unlimited_time(self):
        """Test parsing partition with unlimited time."""
        output = """PartitionName=interactive
   MaxCpusPerNode=64
   MaxTime=UNLIMITED
   PartitionName=interactive"""

        partition = parse_scontrol_partition_output(output)

        assert partition is not None
        assert partition.max_time is None  # UNLIMITED becomes None


class TestFormatLimitsCard:
    """Test limits card formatting."""

    def test_format_card_with_limits(self):
        """Test formatting card with user limits."""
        limits = UserResourceLimits(
            username="testuser",
            account=AccountLimits(
                username="testuser",
                account="default",
                max_cpus_per_user=128,
                max_cpus_running=64,
                max_jobs=10,
                max_jobs_submit=20,
                max_node_per_job=4,
                max_wall_duration="24:00:00",
                grp_cpu_limit=256,
                grp_job_limit=50,
                qos_list=["normal", "high"]
            ),
            qos_limits=[],
            partition_limits=[],
            timestamp="2026-01-22 14:00:00"
        )

        card = format_limits_card(limits, width=70)

        output = "\n".join(card)
        assert "testuser" in output
        assert "default" in output
        assert "128" in output  # Max CPUs
        assert "Max CPUs" in output

    def test_format_card_no_limits(self):
        """Test formatting card when no limits found."""
        limits = UserResourceLimits(
            username="testuser",
            account=None,
            qos_limits=[],
            partition_limits=[],
            timestamp="2026-01-22 14:00:00"
        )

        card = format_limits_card(limits, width=70)

        output = "\n".join(card)
        assert "testuser" in output
        assert "No limits found" in output


class TestFormatLimitsDetailed:
    """Test detailed limits formatting."""

    def test_format_detailed_with_limits(self):
        """Test detailed format with user limits."""
        limits = UserResourceLimits(
            username="testuser",
            account=AccountLimits(
                username="testuser",
                account="default",
                max_cpus_per_user=128,
                max_cpus_running=64,
                max_jobs=10,
                max_jobs_submit=20,
                max_node_per_job=4,
                max_wall_duration="24:00:00",
                grp_cpu_limit=256,
                grp_job_limit=50,
                qos_list=["normal"]
            ),
            qos_limits=[],
            partition_limits=[],
            timestamp="2026-01-22 14:00:00"
        )

        output = format_limits_detailed(limits)

        assert "testuser" in output
        assert "ACCOUNT LIMITS" in output
        assert "Max CPUs per user: 128" in output
        assert "Max concurrent jobs: 10" in output


class TestFormatLimitsCompact:
    """Test compact limits formatting."""

    def test_format_compact_with_limits(self):
        """Test compact format with user limits."""
        limits = UserResourceLimits(
            username="testuser",
            account=AccountLimits(
                username="testuser",
                account="default",
                max_cpus_per_user=128,
                max_cpus_running=64,
                max_jobs=10,
                max_jobs_submit=20,
                max_node_per_job=4,
                max_wall_duration="24:00:00",
                grp_cpu_limit=256,
                grp_job_limit=50,
                qos_list=["normal"]
            ),
            qos_limits=[],
            partition_limits=[],
            timestamp="2026-01-22 14:00:00"
        )

        output = format_limits_compact(limits)

        assert "testuser" in output
        assert "QUICK LIMITS REFERENCE" in output
        assert "128 CPUs" in output
        assert "10 jobs" in output
        assert "24:00:00" in output

    def test_format_compact_no_limits(self):
        """Test compact format with no limits."""
        limits = UserResourceLimits(
            username="testuser",
            account=None,
            qos_limits=[],
            partition_limits=[],
            timestamp="2026-01-22 14:00:00"
        )

        output = format_limits_compact(limits)

        assert "testuser" in output
        assert "No limits found" in output


class TestQoSLimits:
    """Test QoS limits handling."""

    def test_qos_limits_structure(self):
        """Test QoS limits data structure."""
        qos = QoSLimits(
            name="high",
            max_cpus_per_user=256,
            max_jobs_per_user=20,
            max_wall_duration="48:00:00",
            max_nodes_per_job=8,
            priority=100,
            description="High priority QoS"
        )

        assert qos.name == "high"
        assert qos.max_cpus_per_user == 256
        assert qos.max_jobs_per_user == 20
