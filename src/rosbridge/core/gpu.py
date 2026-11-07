"""Match the CUDA device to its Vulkan index, so Isaac renders and simulates on one GPU. No ROS.

Isaac Sim selects its renderer by Vulkan index and physics by CUDA ordinal, and
the two enumerations need not agree on multi-GPU hosts. Discovery creates no
CUDA context and no Vulkan logical device.
"""
import ctypes as c
import os
import uuid


def cuda_uuid(ordinal=0):
    cuda = c.CDLL("libcuda.so.1")

    def check(code):
        if code:
            raise RuntimeError(f"CUDA discovery failed: CUresult {code}")
    check(cuda.cuInit(0))
    device = c.c_int()
    check(cuda.cuDeviceGet(c.byref(device), ordinal))
    identity = (c.c_ubyte * 16)()
    check(cuda.cuDeviceGetUuid_v2(c.byref(identity), device))
    return uuid.UUID(bytes=bytes(identity))


def vulkan_uuids():
    """Vulkan physical devices' UUIDs in the order a current Vulkan loader enumerates them (Kit's bundled one):
    discrete GPUs first, then integrated, virtual and CPU devices, each kind by PCI address. Older system loaders
    (1.3.204) return the driver's own order, which on a multi-GPU host differs from Kit's, so the index must not
    come from the system loader's raw enumeration."""
    class Application(c.Structure):
        _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("name", c.c_char_p),
                    ("version", c.c_uint32), ("engine", c.c_char_p),
                    ("engineVersion", c.c_uint32), ("apiVersion", c.c_uint32)]

    class InstanceInfo(c.Structure):
        _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("flags", c.c_uint32),
                    ("application", c.POINTER(Application)), ("layerCount", c.c_uint32),
                    ("layers", c.c_void_p), ("extensionCount", c.c_uint32),
                    ("extensions", c.c_void_p)]

    class DeviceID(c.Structure):
        _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p),
                    ("deviceUUID", c.c_ubyte * 16), ("driverUUID", c.c_ubyte * 16),
                    ("deviceLUID", c.c_ubyte * 8), ("deviceNodeMask", c.c_uint32),
                    ("deviceLUIDValid", c.c_uint32)]

    class PCIBusInfo(c.Structure):
        _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("domain", c.c_uint32),
                    ("bus", c.c_uint32), ("device", c.c_uint32), ("function", c.c_uint32)]

    class Properties(c.Structure):
        # Output-only VkPhysicalDeviceProperties: reserve more than its size. Its fifth uint32 is the device type.
        _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("properties", c.c_uint32 * 256)]

    vk = c.CDLL("libvulkan.so.1")
    vk.vkCreateInstance.argtypes = [c.POINTER(InstanceInfo), c.c_void_p, c.POINTER(c.c_void_p)]
    vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
    vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_void_p)]
    vk.vkGetPhysicalDeviceProperties2.argtypes = [c.c_void_p, c.POINTER(Properties)]

    def check(code):
        if code:
            raise RuntimeError(f"Vulkan discovery failed: VkResult {code}")
    app = Application(sType=0, name=b"rosbridge-gpu-probe", apiVersion=(1 << 22) | (1 << 12))
    instance = c.c_void_p()
    check(vk.vkCreateInstance(c.byref(InstanceInfo(sType=1, application=c.pointer(app))), None,
                              c.byref(instance)))
    # VkPhysicalDeviceType: other 0, integrated 1, discrete 2, virtual 3, cpu 4
    rank = {2: 0, 1: 1, 3: 2, 4: 3}
    try:
        count = c.c_uint32()
        check(vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None))
        devices = (c.c_void_p * count.value)()
        check(vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices))
        found = []
        for device in devices[:count.value]:
            pci = PCIBusInfo(sType=1000212000)
            identity = DeviceID(sType=1000071004, pNext=c.cast(c.pointer(pci), c.c_void_p))
            properties = Properties(sType=1000059001, pNext=c.cast(c.pointer(identity), c.c_void_p))
            vk.vkGetPhysicalDeviceProperties2(device, c.byref(properties))
            key = (rank.get(properties.properties[4], 4), pci.domain, pci.bus, pci.device, pci.function)
            found.append((key, uuid.UUID(bytes=bytes(identity.deviceUUID))))
        return [identity for _, identity in sorted(found, key=lambda item: item[0])]
    finally:
        vk.vkDestroyInstance(instance, None)


def vulkan_index(ordinal=0):
    """The Vulkan physical-device index of CUDA device `ordinal`."""
    target = cuda_uuid(ordinal)
    matches = [index for index, identity in enumerate(vulkan_uuids()) if identity == target]
    if len(matches) != 1:
        raise RuntimeError(f"CUDA GPU {target} has no unique Vulkan device")
    return matches[0]


def single_gpu():
    """Expose only the first visible CUDA device to this process; returns its Vulkan index.

    Call before CUDA initializes. On IOMMU hosts Kit otherwise validates peer-to-peer
    access between every visible GPU, holding a context on each for minutes.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return vulkan_index(0)
