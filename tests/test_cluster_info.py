"""Tests for cluster resource information module."""

import pytest
from experiment.cluster.info import (
    PartitionInfo,
    QueueInfo,
    ClusterResourceInfo,
    HardwareGroup,
    PartitionGroup,
    parse_sinfo_output,
    parse_squeue_output,
    aggregate_queue_stats,
    aggregate_partitions_by_hardware,
    parse_memory_value,
    format_memory,
    format_compact_info,
    format_detailed_info,
    format_verbose_info,
)


class TestParseMemoryValue:
    """Test memory value parsing."""

    def test_parse_memory_megabytes(self):
        """Test parsing memory in MB."""
        assert parse_memory_value("256") == 256
        assert parse_memory_value("256M") == 256

    def test_parse_memory_gigabytes(self):
        """Test parsing memory in GB."""
        assert parse_memory_value("64G") == 64 * 1024
        assert parse_memory_value("2.5G") == int(2.5 * 1024)

    def test_parse_memory_terabytes(self):
        """Test parsing memory in TB."""
        assert parse_memory_value("1T") == 1024 * 1024
        assert parse_memory_value("2T") == 2 * 1024 * 1024

    def test_parse_memory_kilobytes(self):
        """Test parsing memory in KB."""
        assert parse_memory_value("1024K") == 1


class TestFormatMemory:
    """Test memory formatting."""

    def test_format_memory_megabytes(self):
        """Test formatting memory in MB."""
        assert format_memory(256) == "256MB"
        assert format_memory(512) == "512MB"

    def test_format_memory_gigabytes(self):
        """Test formatting memory in GB."""
        assert format_memory(1024) == "1GB"
        assert format_memory(64 * 1024) == "64GB"
        assert format_memory(256 * 1024) == "256GB"


class TestParseSinfoOutput:
    """Test sinfo output parsing."""

    def test_parse_sinfo_single_partition(self):
        """Test parsing single partition from sinfo."""
        sinfo_output = "gpu|up|12:00:00|10|idle|0/800/800|256G|gpu:4|gpu[0-9]"

        partitions = parse_sinfo_output(sinfo_output)

        assert len(partitions) == 1
        partition = partitions[0]
        assert partition.name == "gpu"
        assert partition.availability == "up"
        assert partition.time_limit == "12:00:00"
        assert partition.num_nodes == 10
        assert partition.node_state == "idle"
        assert partition.cpus_allocated == 0
        assert partition.cpus_idle == 800
        assert partition.cpus_total == 800
        assert partition.memory_mb == 256 * 1024
        assert partition.gpus == "gpu:4"

    def test_parse_sinfo_multiple_partitions(self):
        """Test parsing multiple partitions."""
        sinfo_output = """gpu|up|12:00:00|10|idle|0/800/800|256G|gpu:4|gpu[0-9]
standard|up|48:00:00|50|mixed|1200/1800/3000|64G|N/A|std[0-49]
debug|up|00:30:00|2|alloc|8/24/32|32G|N/A|debug[0-1]"""

        partitions = parse_sinfo_output(sinfo_output)

        assert len(partitions) == 3
        assert partitions[0].name == "gpu"
        assert partitions[1].name == "standard"
        assert partitions[2].name == "debug"
        assert partitions[2].cpus_allocated == 8

    def test_parse_sinfo_malformed_lines(self):
        """Test parsing with malformed lines."""
        sinfo_output = """gpu|up|12:00:00|10|idle|0/800/800|256G|gpu:4|gpu[0-9]
invalid_line_missing_fields
standard|up|48:00:00|50|mixed|1200/1800/3000|64G|N/A|std[0-49]"""

        partitions = parse_sinfo_output(sinfo_output)

        assert len(partitions) == 2
        assert partitions[0].name == "gpu"
        assert partitions[1].name == "standard"

    def test_parse_sinfo_down_partition(self):
        """Test parsing down/drain partitions."""
        sinfo_output = "down_part|down|unlimited|5|down|0/0/0|128G|N/A|node[0-4]"

        partitions = parse_sinfo_output(sinfo_output)

        assert len(partitions) == 1
        assert partitions[0].name == "down_part"
        assert partitions[0].availability == "down"
        assert partitions[0].node_state == "down"

    def test_parse_sinfo_unlimited_time(self):
        """Test parsing unlimited time limit."""
        sinfo_output = "batch|up|unlimited|20|idle|0/400/400|64G|N/A|batch[0-19]"

        partitions = parse_sinfo_output(sinfo_output)

        assert len(partitions) == 1
        assert partitions[0].time_limit == "unlimited"


class TestParseSqueueOutput:
    """Test squeue output parsing."""

    def test_parse_squeue_single_job(self):
        """Test parsing single job from squeue."""
        squeue_output = "12345|gpu|RUNNING|1|4"

        jobs = parse_squeue_output(squeue_output)

        assert len(jobs) == 1
        partition, state, num_nodes, num_cpus = jobs[0]
        assert partition == "gpu"
        assert state == "RUNNING"
        assert num_nodes == 1
        assert num_cpus == 4

    def test_parse_squeue_multiple_jobs(self):
        """Test parsing multiple jobs."""
        squeue_output = """12345|gpu|RUNNING|1|4
12346|gpu|PENDING|2|8
12347|standard|RUNNING|4|16
12348|standard|PENDING|1|2"""

        jobs = parse_squeue_output(squeue_output)

        assert len(jobs) == 4
        assert jobs[0][1] == "RUNNING"
        assert jobs[1][1] == "PENDING"

    def test_parse_squeue_empty(self):
        """Test parsing empty queue."""
        squeue_output = ""

        jobs = parse_squeue_output(squeue_output)

        assert len(jobs) == 0

    def test_parse_squeue_malformed(self):
        """Test parsing with malformed lines."""
        squeue_output = """12345|gpu|RUNNING|1|4
invalid_line
12346|gpu|PENDING|2|8"""

        jobs = parse_squeue_output(squeue_output)

        assert len(jobs) == 2


class TestAggregateQueueStats:
    """Test queue statistics aggregation."""

    def test_aggregate_single_partition(self):
        """Test aggregating stats for single partition."""
        jobs = [
            ("gpu", "RUNNING", 1, 4),
            ("gpu", "RUNNING", 2, 8),
            ("gpu", "PENDING", 1, 2),
        ]

        stats = aggregate_queue_stats(jobs)

        assert len(stats) == 1
        assert stats[0].partition == "gpu"
        assert stats[0].running_jobs == 2
        assert stats[0].pending_jobs == 1
        assert stats[0].running_cpus == 12
        assert stats[0].pending_cpus == 2

    def test_aggregate_multiple_partitions(self):
        """Test aggregating stats for multiple partitions."""
        jobs = [
            ("gpu", "RUNNING", 1, 4),
            ("gpu", "PENDING", 1, 2),
            ("standard", "RUNNING", 4, 16),
            ("standard", "PENDING", 2, 4),
        ]

        stats = aggregate_queue_stats(jobs)

        assert len(stats) == 2
        gpu_stats = next(s for s in stats if s.partition == "gpu")
        std_stats = next(s for s in stats if s.partition == "standard")

        assert gpu_stats.running_jobs == 1
        assert gpu_stats.pending_jobs == 1
        assert std_stats.running_jobs == 1
        assert std_stats.pending_jobs == 1

    def test_aggregate_empty_queue(self):
        """Test aggregating empty queue."""
        jobs = []

        stats = aggregate_queue_stats(jobs)

        assert len(stats) == 0

    def test_aggregate_other_states(self):
        """Test that other states are ignored."""
        jobs = [
            ("gpu", "RUNNING", 1, 4),
            ("gpu", "COMPLETED", 1, 4),
            ("gpu", "FAILED", 1, 4),
            ("gpu", "PENDING", 1, 2),
        ]

        stats = aggregate_queue_stats(jobs)

        assert len(stats) == 1
        assert stats[0].running_jobs == 1
        assert stats[0].pending_jobs == 1
        # Only PENDING and RUNNING should be counted


class TestAggregatePartitionsByHardware:
    """Test partition aggregation by hardware and node state."""

    def test_aggregate_homogeneous_partition(self):
        """Test aggregating homogeneous partition (all same hardware)."""
        partitions = [
            PartitionInfo("cpu", "up", "48:00:00", 7, "mixed", 116, 332, 448, 187*1024, "N/A", "node[0-6]"),
            PartitionInfo("cpu", "up", "48:00:00", 5, "allocated", 320, 0, 320, 187*1024, "N/A", "node[7-11]"),
            PartitionInfo("cpu", "up", "48:00:00", 8, "idle", 0, 512, 512, 187*1024, "N/A", "node[12-19]"),
        ]

        groups = aggregate_partitions_by_hardware(partitions)

        assert len(groups) == 1
        assert groups[0].name == "cpu"
        # Should have only 1 hardware group (all same: 64 CPU/node, 187GB, no GPU)
        assert len(groups[0].hardware_groups) == 1
        # Should have 3 node states
        hw = groups[0].hardware_groups[0]
        assert len(hw.node_states) == 3
        assert "mixed" in hw.node_states
        assert "allocated" in hw.node_states
        assert "idle" in hw.node_states

    def test_aggregate_heterogeneous_partition(self):
        """Test aggregating heterogeneous partition (different GPU types)."""
        partitions = [
            PartitionInfo("gpu", "up", "2-00:00:00", 1, "idle", 4, 28, 32, 187*1024, "gpu:v100:2", "gpu0"),
            PartitionInfo("gpu", "up", "2-00:00:00", 2, "mixed", 80, 48, 128, 188*1024, "gpu:rtx6000:3", "gpu[1-2]"),
            PartitionInfo("gpu", "up", "2-00:00:00", 3, "idle", 154, 230, 384, 503*1024, "gpu:h100:4", "gpu[3-5]"),
        ]

        groups = aggregate_partitions_by_hardware(partitions)

        assert len(groups) == 1
        assert groups[0].name == "gpu"
        # Should have 3 hardware groups (v100, rtx6000, h100)
        assert len(groups[0].hardware_groups) == 3

    def test_aggregate_multiple_partitions(self):
        """Test aggregating multiple different partitions."""
        partitions = [
            PartitionInfo("cpu", "up", "48:00:00", 7, "mixed", 116, 332, 448, 187*1024, "N/A", "node[0-6]"),
            PartitionInfo("gpu", "up", "2-00:00:00", 1, "idle", 4, 28, 32, 187*1024, "gpu:v100:2", "gpu0"),
        ]

        groups = aggregate_partitions_by_hardware(partitions)

        assert len(groups) == 2
        names = {g.name for g in groups}
        assert "cpu" in names
        assert "gpu" in names


class TestFormatCompactInfo:
    """Test compact format output."""

    def test_format_compact_single_partition(self):
        """Test formatting single partition in compact mode."""
        partition = PartitionInfo(
            name="gpu",
            availability="up",
            time_limit="12:00:00",
            num_nodes=10,
            node_state="idle",
            cpus_allocated=320,
            cpus_idle=480,
            cpus_total=800,
            memory_mb=256 * 1024,
            gpus="gpu:4",
            node_list="gpu[0-9]"
        )

        hw_group = HardwareGroup(
            gpu_type="gpu:4",
            cpus_per_node=80,
            memory_mb=256 * 1024,
            node_states={"idle": (10, 320, 480)}
        )
        partition_group = PartitionGroup(
            name="gpu",
            availability="up",
            time_limit="12:00:00",
            hardware_groups=[hw_group]
        )

        queue_info = QueueInfo(
            partition="gpu",
            pending_jobs=12,
            running_jobs=8,
            pending_cpus=192,
            running_cpus=320
        )

        info = ClusterResourceInfo(
            partitions=[partition],
            partition_groups=[partition_group],
            queue_stats=[queue_info],
            timestamp="2026-01-22 14:35:42"
        )

        output = format_compact_info(info, "cluster.example.edu", "user")

        assert "cluster.example.edu" in output
        assert "1 partitions" in output
        assert "gpu" in output
        assert "Queue: 12 pending | 8 running" in output

    def test_format_compact_multiple_partitions(self):
        """Test formatting multiple partitions."""
        partitions = [
            PartitionInfo("gpu", "up", "12:00:00", 10, "idle", 320, 480, 800, 256*1024, "gpu:4", "gpu[0-9]"),
            PartitionInfo("standard", "up", "48:00:00", 50, "mixed", 1200, 1800, 3000, 64*1024, "N/A", "std[0-49]"),
        ]

        gpu_hw = HardwareGroup(
            gpu_type="gpu:4",
            cpus_per_node=80,
            memory_mb=256*1024,
            node_states={"idle": (10, 320, 480)}
        )
        gpu_group = PartitionGroup("gpu", "up", "12:00:00", [gpu_hw])

        std_hw = HardwareGroup(
            gpu_type="N/A",
            cpus_per_node=60,
            memory_mb=64*1024,
            node_states={"mixed": (50, 1200, 1800)}
        )
        std_group = PartitionGroup("standard", "up", "48:00:00", [std_hw])

        queue_stats = [
            QueueInfo("gpu", 12, 8, 192, 320),
            QueueInfo("standard", 3, 27, 48, 1080),
        ]

        info = ClusterResourceInfo(
            partitions=partitions,
            partition_groups=[gpu_group, std_group],
            queue_stats=queue_stats,
            timestamp="2026-01-22 14:35:42"
        )

        output = format_compact_info(info, "cluster.example.edu", "user")

        assert "2 partitions" in output
        assert "gpu" in output
        assert "standard" in output

    def test_format_compact_empty_queue(self):
        """Test formatting with empty queue."""
        partition = PartitionInfo(
            name="gpu",
            availability="up",
            time_limit="12:00:00",
            num_nodes=10,
            node_state="idle",
            cpus_allocated=0,
            cpus_idle=800,
            cpus_total=800,
            memory_mb=256 * 1024,
            gpus="gpu:4",
            node_list="gpu[0-9]"
        )

        hw_group = HardwareGroup(
            gpu_type="gpu:4",
            cpus_per_node=80,
            memory_mb=256 * 1024,
            node_states={"idle": (10, 0, 800)}
        )
        partition_group = PartitionGroup("gpu", "up", "12:00:00", [hw_group])

        info = ClusterResourceInfo(
            partitions=[partition],
            partition_groups=[partition_group],
            queue_stats=[],
            timestamp="2026-01-22 14:35:42"
        )

        output = format_compact_info(info, "cluster.example.edu", "user")

        assert "Queue: 0 pending | 0 running" in output


class TestFormatDetailedInfo:
    """Test detailed format output."""

    def test_format_detailed_single_partition(self):
        """Test formatting single partition in detailed mode."""
        partition = PartitionInfo(
            name="gpu",
            availability="up",
            time_limit="12:00:00",
            num_nodes=10,
            node_state="idle",
            cpus_allocated=320,
            cpus_idle=480,
            cpus_total=800,
            memory_mb=256 * 1024,
            gpus="gpu:4",
            node_list="gpu[0-9]"
        )

        hw_group = HardwareGroup(
            gpu_type="gpu:4",
            cpus_per_node=80,
            memory_mb=256 * 1024,
            node_states={"idle": (10, 320, 480)}
        )
        partition_group = PartitionGroup("gpu", "up", "12:00:00", [hw_group])

        queue_info = QueueInfo(
            partition="gpu",
            pending_jobs=12,
            running_jobs=8,
            pending_cpus=192,
            running_cpus=320
        )

        info = ClusterResourceInfo(
            partitions=[partition],
            partition_groups=[partition_group],
            queue_stats=[queue_info],
            timestamp="2026-01-22 14:35:42"
        )

        output = format_detailed_info(info, "cluster.example.edu", "user")

        assert "Cluster: cluster.example.edu" in output
        assert "RESOURCE SUMMARY" in output
        assert "gpu" in output
        assert "Status: up" in output


class TestFormatVerboseInfo:
    """Test verbose format output."""

    def test_format_verbose_includes_detailed(self):
        """Test that verbose format includes detailed output."""
        partition = PartitionInfo(
            name="gpu",
            availability="up",
            time_limit="12:00:00",
            num_nodes=10,
            node_state="idle",
            cpus_allocated=320,
            cpus_idle=480,
            cpus_total=800,
            memory_mb=256 * 1024,
            gpus="gpu:4",
            node_list="gpu[0-9]"
        )

        hw_group = HardwareGroup(
            gpu_type="gpu:4",
            cpus_per_node=80,
            memory_mb=256 * 1024,
            node_states={"idle": (10, 320, 480)}
        )
        partition_group = PartitionGroup("gpu", "up", "12:00:00", [hw_group])

        queue_info = QueueInfo(
            partition="gpu",
            pending_jobs=12,
            running_jobs=8,
            pending_cpus=192,
            running_cpus=320
        )

        info = ClusterResourceInfo(
            partitions=[partition],
            partition_groups=[partition_group],
            queue_stats=[queue_info],
            timestamp="2026-01-22 14:35:42"
        )

        output = format_verbose_info(info, "cluster.example.edu", "user")

        # Should include detailed content
        assert "RESOURCE SUMMARY" in output
        assert "gpu" in output

        # Should include hardware details
        assert "Hardware:" in output
        assert "CPUs per node:" in output


class TestEmptyResults:
    """Test handling of empty or no results."""

    def test_format_compact_no_partitions(self):
        """Test compact format with no partitions."""
        info = ClusterResourceInfo(partitions=[], partition_groups=[], queue_stats=[], timestamp="2026-01-22 14:35:42")

        output = format_compact_info(info, "cluster.example.edu", "user")

        assert "No partitions found" in output

    def test_format_detailed_no_partitions(self):
        """Test detailed format with no partitions."""
        info = ClusterResourceInfo(partitions=[], partition_groups=[], queue_stats=[], timestamp="2026-01-22 14:35:42")

        output = format_detailed_info(info, "cluster.example.edu", "user")

        assert "No partitions found" in output

    def test_format_verbose_no_partitions(self):
        """Test verbose format with no partitions."""
        info = ClusterResourceInfo(partitions=[], partition_groups=[], queue_stats=[], timestamp="2026-01-22 14:35:42")

        output = format_verbose_info(info, "cluster.example.edu", "user")

        assert "No partitions found" in output
