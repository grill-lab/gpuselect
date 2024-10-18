import random
from typing import Generator

import pytest

from .context import gpuselect
from gpuselect import GpuInfo, __filter_gpus

GPU_NAME_3090 = "NVIDIA GeForce RTX 3090"
GPU_NAME_RTX_6000_ADA = "NVIDIA RTX 6000 Ada Generation"


@pytest.fixture
def sample_gpuinfo() -> Generator:
    """Factory method for creating populated GpuInfo instances."""

    def _build_gpuinfo(
        device: int,
        name: str = "Test GPU",
        util: int = 0,
        mem_util: int = 0,
        processes: int = 0,
    ) -> GpuInfo:
        return GpuInfo(
            device=device, name=name, util=util, mem_util=mem_util, processes=processes
        )

    yield _build_gpuinfo


def test_filter_gpus_name(sample_gpuinfo: Generator) -> None:
    """
    Test for selecting a single GPU of a particular type
    """
    gpu_info = [sample_gpuinfo(device=d, name=GPU_NAME_3090) for d in range(5)]

    partial_name = "3090"
    filtered_gpus = __filter_gpus(
        gpu_info,
        count=1,
        devices=[],
        name=partial_name,
        util=0,
        mem_util=0,
        processes=0,
        selector=None,
    )

    assert len(filtered_gpus) == 1 and partial_name in filtered_gpus[0].name

    full_name = GPU_NAME_3090

    filtered_gpus = __filter_gpus(
        gpu_info,
        count=1,
        devices=[],
        name=full_name,
        util=0,
        mem_util=0,
        processes=0,
        selector=None,
    )

    assert len(filtered_gpus) == 1 and full_name == filtered_gpus[0].name


def test_filter_gpus_device_id(sample_gpuinfo: Generator) -> None:
    """
    Test for selecting GPUs by device ID
    """
    gpu_info = [sample_gpuinfo(device=d, name=GPU_NAME_3090) for d in range(5)]

    device_ids = [1, 3]
    filtered_gpus = __filter_gpus(
        gpu_info,
        count=2,
        devices=device_ids,
        name=None,
        util=0,
        mem_util=0,
        processes=0,
        selector=None,
    )

    assert len(filtered_gpus) == len(device_ids)
    filtered_ids = [gpu.device for gpu in filtered_gpus]
    for id in device_ids:
        assert id in filtered_ids


def test_filter_gpus_utilization(sample_gpuinfo: Generator) -> None:
    """
    Test that GPUs are not selected if they're above the utilization threshold
    """
    gpu_info = [
        sample_gpuinfo(device=d, name=GPU_NAME_3090, util=random.randint(5, 50))
        for d in range(5)
    ]

    # request any GPU with sufficiently low utilization/process count
    filtered_gpus = __filter_gpus(
        gpu_info,
        count=1,
        devices=[],
        name=None,
        util=0,
        mem_util=0,
        processes=0,
        selector=None,
    )

    assert len(filtered_gpus) == 0


def test_filter_gpus_mem_utilization(sample_gpuinfo: Generator) -> None:
    """
    Test that GPUs are not selected if they're above the memory utilization threshold
    """
    gpu_info = [
        sample_gpuinfo(device=d, name=GPU_NAME_3090, mem_util=random.randint(5, 50))
        for d in range(5)
    ]

    # request any GPU with sufficiently low utilization/process count
    filtered_gpus = __filter_gpus(
        gpu_info,
        count=1,
        devices=[],
        name=None,
        util=0,
        mem_util=0,
        processes=0,
        selector=None,
    )

    assert len(filtered_gpus) == 0


def test_filter_gpus_process_count(sample_gpuinfo: Generator) -> None:
    """
    Test that GPUs are not selected if they're above the process count threshold
    """
    gpu_info = [
        sample_gpuinfo(device=d, name=GPU_NAME_3090, processes=random.randint(1, 10))
        for d in range(5)
    ]

    # request any GPU with sufficiently low utilization/process count
    filtered_gpus = __filter_gpus(
        gpu_info,
        count=1,
        devices=[],
        name=None,
        util=0,
        mem_util=0,
        processes=0,
        selector=None,
    )

    assert len(filtered_gpus) == 0


def test_filter_gpus_selector(sample_gpuinfo: Generator) -> None:
    """
    Test using a custom filter function
    """

    # define a custom filter that will accept any GPU with "3090" in the name,
    # 0% process utilization, and at most 1 process
    def custom_filter(gpu: GpuInfo) -> bool:
        return "3090" in gpu.name and gpu.util == 0 and gpu.processes <= 1

    # create a GpuInfo that should match this filter
    gpu_info = [
        sample_gpuinfo(device=0, name=GPU_NAME_3090, util=0, mem_util=0, processes=1)
    ]
    # add some other GpuInfos that shouldn't match
    gpu_info.extend(
        sample_gpuinfo(
            device=d + 1,
            name=GPU_NAME_3090,
            util=10,
            mem_util=0,
            processes=random.randint(1, 10),
        )
        for d in range(5)
    )

    # request any GPU with sufficiently low utilization/process count
    filtered_gpus = __filter_gpus(
        gpu_info,
        count=1,
        devices=[],
        name=None,
        util=0,
        mem_util=0,
        processes=0,
        selector=custom_filter,
    )

    assert len(filtered_gpus) == 1
