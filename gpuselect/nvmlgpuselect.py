from __future__ import annotations  # for 3.8/3.9
import os
import argparse
import random
import logging
import subprocess
import sys
import select
from typing import NamedTuple, Callable

import pynvml

logger = logging.getLogger("gpuselect")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))


class GpuInfo(NamedTuple):
    device: int
    name: str
    util: int
    mem_util: int
    mem_free: int
    mem_used: int
    mem_total: int
    processes: int
    fan_speed: int
    effective_power_limt: int
    performance_state: int
    power_usage: int
    temperature: int
    is_throttling: int


def pynvml_get_gpu_state(device_index: int) -> GpuInfo:
    # note that all of these functions typically return a wrapper around a C struct, see
    # the API for details:
    # https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html#nvml-api-reference
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

    device = device_index
    name = pynvml.nvmlDeviceGetName(handle)
    mem_free = mem_info.free
    mem_used = mem_info.used
    mem_total = mem_info.total
    # note this is just the *number* of processes
    processes = len(pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle))

    # fan speed, effective power limit, performance state, power usage, temperature
    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
    effective_power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
    performance_state = pynvml.nvmlDeviceGetPerformanceState(handle)
    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

    # try to check for throttling. the result is a bitmask and the values are listed at:
    # https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksEventReasons.html#group__nvmlClocksEventReasons_1g78486f43e3612308528ec9849b9658ec
    #
    # since there seem to be many values that indicate different types of throttling and all we
    # really want is to know IF there is any throttling, this just checks for a couple of non-throttling states:
    #   - nvmlClocksEventReasonGpuIdle (GPU is idle, clocks are low for that reason)
    #   - nvmlClocksEventReasonNone (GPU is clocked as high as possible)
    # if either of these are set in the bitmask then we shouldn't be throttling
    reasons = pynvml.nvmlDeviceGetCurrentClocksEventReasons(handle)

    is_throttling = (
        0
        if reasons & pynvml.nvmlClocksEventReasonGpuIdle
        or reasons & pynvml.nvmlClocksEventReasonNone
        else 1
    )

    return GpuInfo(
        device,
        name,
        utilization.gpu,
        utilization.memory,
        mem_free,
        mem_used,
        mem_total,
        processes,
        fan_speed,
        effective_power_limit,
        performance_state,
        power_usage,
        temperature,
        is_throttling,
    )


def __scan_gpus(devices: list[int]) -> list[GpuInfo]:
    """
    Return a list of available GPUs and their current utilization stats.

    This method enumerates GPUs through the NVML API, and for each one creates
    a `GpuInfo` object which contains the following information (field names in parentheses):
        - device ID (device)
        - device name (name)
        - utilization % (util)
        - memory utilization % (mem_util)
        - memory free, bytes (mem_free)
        - memory used, bytes (mem_used)
        - memory total, bytes (mem_total)
        - process count (processes)
        - fan speed % (fan_speed)
        - effective power limit in mW (effective_power_limit)
        - performance state, 0 = max, 15 = min (performance_state)
        - power usage in mW (power_usage)
        - temperature in deg C (temperature)
        - throttling state, 0 if no throttling, for other values see https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksEventReasons.html (is_throttling)

    Args:
        devices: a list of device IDs that may have been selected by the application using
            this module. if this list is populated the IDs are checked for validity. if
            the list is empty no checks are performed.

    Returns: a list of GpuInfo objects
    """
    gpu_info: list[GpuInfo] = []

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # or FASTEST_FIRST
    pynvml.nvmlInit()
    device_count: int = pynvml.nvmlDeviceGetCount()

    logger.debug(f"GPU count: {device_count}")

    for d in devices:
        if d < 0 or d >= device_count:
            logger.debug(
                f"Device ID {d} is outside the expected range of 0 - {device_count - 1}"
            )
            pynvml.nvmlShutdown()
            raise Exception(f"Invalid device ID {d}")

    # iterate over all GPUs in the system
    for i in range(device_count):
        info = pynvml_get_gpu_state(i)
        gpu_info.append(info)
        logger.debug(f"Device {i}: {gpu_info[-1]}")

    pynvml.nvmlShutdown()

    return gpu_info


def __filter_gpus(
    gpu_info: list[GpuInfo],
    count: int,
    devices: list[int],
    name: str | None,
    util: int,
    mem_util: int,
    processes: int,
    selector: Callable[[GpuInfo], bool] | None,
) -> list[GpuInfo]:
    """
    Filter list of available GPUs and return selected device(s).
    """

    filtered_gpus: list[GpuInfo] = []

    # user has provided a filter function, this overrides the standard filters.
    if selector is not None:
        logger.debug(f"Applying selector to {len(gpu_info)} devices, count={count}")

        filtered_gpus = list(filter(selector, gpu_info))
    else:
        # First filter on device ID OR name (mutually exclusive)
        if len(devices) > 0:
            filtered_gpus = [gpu_info[d] for d in devices]
            logger.debug(f"Device filter, {len(filtered_gpus)} devices left")
        elif name is not None:
            # match any substring in the device name
            matched_ids: list[int] = []
            for gpu in gpu_info:
                if name in gpu.name:
                    matched_ids.append(gpu.device)
                    logger.debug(f"Name match of {name} on {gpu}")

            filtered_gpus = [gpu_info[d] for d in matched_ids]
            logger.debug(f"Name filter, {len(filtered_gpus)} devices left")
        else:
            filtered_gpus = gpu_info

        # Then filter by utilization/process count
        before_util = len(filtered_gpus)
        filtered_gpus = list(filter(lambda info: info.util <= util, filtered_gpus))
        logger.debug(
            f"Utilization filter matched {len(filtered_gpus)} / {before_util} GPUs"
        )

        before_mem_util = len(filtered_gpus)
        filtered_gpus = list(
            filter(lambda info: info.mem_util <= mem_util, filtered_gpus)
        )
        logger.debug(
            f"Memory utilization filter matched {len(filtered_gpus)} / {before_mem_util} GPUs"
        )

        before_processes = len(filtered_gpus)
        filtered_gpus = list(
            filter(lambda info: info.processes <= processes, filtered_gpus)
        )
        logger.debug(
            f"Processes filter matched {len(filtered_gpus)} / {before_processes} GPUs"
        )

    logger.debug(f"GPUs left after filtering ({count} requested): {len(filtered_gpus)}")
    # Finally prune the list down to <count> devices (and check if sufficient devices remain)
    if count > len(filtered_gpus):
        # TODO: return empty list here instead?
        return filtered_gpus

    # randomly select devices from the available set
    return random.sample(filtered_gpus, count)


def gpustatus(only_cvd: bool = True) -> list[GpuInfo]:
    """
    Utility method to query information about GPUs in the system.

    This method can be used to periodically query GPU utilization/performance metrics. If
    the `only_cvd` parameter is `True`, it will respect the current value of the
    `CUDA_VISIBLE_DEVICES` environment variable and only report on those devices. Otherwise
    it will query all GPUs visible through the NVML API.

    The return value will be a list of `GpuInfo` objects, each of
    which contains the following information (field names in parentheses):
        - device ID (device)
        - device name (name)
        - utilization % (util)
        - memory utilization % (mem_util)
        - memory free, bytes (mem_free)
        - memory used, bytes (mem_used)
        - memory total, bytes (mem_total)
        - process count (processes)
        - fan speed % (fan_speed)
        - effective power limit in mW (effective_power_limit)
        - performance state, 0 = max, 15 = min (performance_state)
        - power usage in mW (power_usage)
        - temperature in deg C (temperature)
        - throttling state, 0 if no throttling, for other values see https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksEventReasons.html (is_throttling)

    Args:
        only_cvd: `True` to query GPUs based on `CUDA_VISIBLE_DEVICES`, `False` to query all

    Returns:
        a list of `GpuInfos`, containing the current value of each metric listed above for each GPU
    """
    gpus: list[GpuInfo] = []
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()

    cvd = os.environ["CUDA_VISIBLE_DEVICES"]
    if len(cvd) == 0 or cvd == "all":
        visible_gpus = list(range(device_count))
    else:
        # CUDA_VISIBLE_DEVICES can also contain comma-separated GPU UUIDs instead of
        # device IDs, but this isn't currently supported here
        visible_gpus = list(map(int, cvd.strip().split(",")))

    # skip GPUs excluded by CUDA_VISIBLE_DEVICES
    for i in range(device_count):
        if i not in visible_gpus:
            continue

        info = pynvml_get_gpu_state(i)
        gpus.append(info)

    pynvml.nvmlShutdown()
    return gpus


def gpuselect(
    count: int = 1,
    devices: int | list[int] | None = None,
    name: str | None = None,
    util: int = 0,
    mem_util: int = 0,
    processes: int = 0,
    selector: Callable[[GpuInfo], bool] | None = None,
    silent: bool = False,
    set_cvd: bool = True,
) -> str:
    """
    Select 1 or more GPUs by updating `CUDA_VISIBLE_DEVICES` based on some simple filters.

    Examples:
        Request a single GPU with "A6000" in the name that has no utilization/processes/memory usage
            > gpuselect(name="A6000")

        Request the GPU with ID=0
            > gpuselect(device=0)

        Request 2 GPUs with names partially matching "A6000"
            > gpuselect(name="A6000", count=2)

        Request an A6000 with some existing utilization allowed
            > gpuselect(name="A6000", util=5, mem=5)

        Request any GPU that has zero utilization
            > gpuselect.select()

    Args:
        count: number of GPUs required (defaults to 1)
        devices: used to select specific devices if required (can be None, a single ID or a list of IDs, defaults to `None`)
        name: used to match GPUs by full/partial string matching in device names (defaults to `None`)
        util: maximum device utilization threshold as a percentage (defaults to 0)
        mem_util: maximum memory utilization threshold as a percentage (defaults to 0)
        processes: maximum process count (defaults to 0)
        selector: a `Callable` object which should take an argument of type `GpuInfo` and return `True`/`False` to indicate
            if the GPU should be included or excluded
        silent: if `False`, an exception is thrown if there are no GPUs matching the selected filters (defaults to `False`)
        set_cvd: if `True`, update the value of `CUDA_VISIBLE_DEVICES` for the current process (defaults to `True`)

    Returns:
    """
    logger.debug(
        f"count={count},devices={devices},name={name},util={util},mem_util={mem_util},processes={processes},silent={silent}"
    )

    # validate parameters

    # count should be >= 1
    if count < 1:
        raise Exception("GPU count must be >= 1")

    # devices should always be a list of ints after this point (possibly empty)
    if devices is not None:
        if isinstance(devices, int):
            devices = [devices]
    else:
        devices = []

    # device and name are mutually exclusive
    if len(devices) > 0 and name is not None:
        raise Exception("Use only one of 'devices' and 'name' parameters")

    # utilization and process count values should all be >= 0
    if util < 0 or mem_util < 0 or processes < 0:
        raise Exception("Utilization/process count thresholds must be >= 0")

    # if this happens, assume the user wants len(devices) GPUs and override count
    if len(devices) > count:
        logger.warning(
            f"count={count} but {len(devices)} device IDs were specified, setting count={len(devices)}"
        )

    gpu_info: list[GpuInfo] = __scan_gpus(devices)

    filtered_gpus = __filter_gpus(
        gpu_info, count, devices, name, util, mem_util, processes, selector
    )

    if len(filtered_gpus) < count:
        msg = f"Requested {count} GPUs, but {len(filtered_gpus)} matched filters"
        logger.debug(msg)
        if silent:
            return ""
        raise Exception(msg)

    # update CUDA_VISIBLE_DEVICES with the list of filtered device IDs
    gpu_ids = [info.device for info in filtered_gpus]
    cuda_visible_devices = ",".join(f"{d}" for d in gpu_ids)
    if set_cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    logger.debug(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

    return cuda_visible_devices


def _int_or_list(arg: str) -> list[int] | int:
    """
    Checks if arg can be parsed as a single int or a comma-separated list

    This is used as an argparse helper below.

    Args:
        arg: an argument string that may contain either a single int or a comma-separated list of them

    Returns:
        parsed argument value (or throws an exception)
    """

    # check if all characters are digits
    if arg.isdigit():
        return int(arg)

    try:
        return [int(n) for n in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Parameter must be a single int or 'int1,int2,...'"
        )


def main():
    parser = argparse.ArgumentParser("GPU selection")
    _ = parser.add_argument(
        "-d",
        "--devices",
        help="Filter GPUs by device ID",
        type=_int_or_list,
        required=False,
    )
    _ = parser.add_argument(
        "-n",
        "--name",
        help="Filter GPUs by substring match on their names",
        type=str,
        required=False,
    )
    _ = parser.add_argument(
        "-u",
        "--util",
        help="Filter GPUs by max utilization percentage",
        type=int,
        required=False,
        default=0,
    )
    _ = parser.add_argument(
        "-m",
        "--mem_util",
        help="Filter GPUs by max memory utilization percentage",
        type=int,
        required=False,
        default=0,
    )
    _ = parser.add_argument(
        "-c",
        "--count",
        help="Number of GPUs that must match other filters",
        type=int,
        required=False,
        default=1,
    )
    _ = parser.add_argument(
        "-p",
        "--processes",
        help="Filter GPUs by max process count",
        type=int,
        required=False,
        default=0,
    )
    _ = parser.add_argument(
        "-q",
        "--quiet",
        help="Do *not* print the new value of CUDA_VISIBLE_DEVICES on exit",
        action="store_true",
    )

    args, unknown_args = parser.parse_known_args()

    cuda_visible_devices = gpuselect(
        args.count, args.devices, args.name, args.util, args.mem_util, args.processes
    )

    if len(unknown_args) == 0:
        if not args.quiet:
            print(cuda_visible_devices)
        sys.exit(0)

    if "--" in unknown_args:
        unknown_args.remove("--")

    print(f"Running: {unknown_args}")
    env = os.environ.copy()
    # env["PYTHONUNBUFFERED"] = 1
    cmd = subprocess.Popen(
        unknown_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )

    # TODO probably needs more testing
    while True:
        reads = [cmd.stdout.fileno(), cmd.stderr.fileno()]
        read_fds = select.select(reads, [], [])

        for fd in read_fds:
            if fd == cmd.stdout.fileno():
                output = cmd.stdout.readline()
                if output is not None:
                    print(output.strip())
            elif fd == cmd.stderr.fileno():
                output = cmd.stderr.readline()
                if output is not None:
                    print(output.strip(), file=sys.stderr)

        if cmd.poll() is not None:
            break

    for output in cmd.stdout:
        print(output.strip())
    for output in cmd.stderr:
        print(output.strip(), file=sys.stderr)


if __name__ == "__main__":
    main()
