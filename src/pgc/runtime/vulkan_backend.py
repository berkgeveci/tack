"""PGC Vulkan backend — compiles kernels via SPIR-V and dispatches via Vulkan compute.

Pipeline:
    PGC IR → SPIR-V binary → vkCreateShaderModule → compute pipeline → vkCmdDispatch

Fields use host-visible coherent memory (zero-copy on integrated GPUs, mapped on discrete).
No per-dispatch copies — data stays on the GPU between kernel calls.

Requires libvulkan.so (Linux), libvulkan.dylib (macOS/MoltenVK), or vulkan-1.dll (Windows).
No Python Vulkan packages needed — uses ctypes directly.
"""

import ctypes
import ctypes.util
import platform
import struct

import numpy as np

from pgc.lang import ir
from pgc.lang.field import Field, DeviceBuffer
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
from pgc.codegen.spirv_gen import generate_spirv

# ---------------------------------------------------------------------------
# Numpy dtype mapping
# ---------------------------------------------------------------------------
_NUMPY_DTYPE = {
    f32: np.float32, f64: np.float64,
    i32: np.int32, i64: np.int64,
    u32: np.uint32, u64: np.uint64,
}

# ---------------------------------------------------------------------------
# Vulkan constants
# ---------------------------------------------------------------------------
VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
VK_STRUCTURE_TYPE_SUBMIT_INFO = 4
VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12
VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16
VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29
VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30
VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18
VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32
VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33
VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34
VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35
VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39
VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40
VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42

VK_API_VERSION_1_1 = (1 << 22) | (1 << 12)  # Vulkan 1.1

VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020
VK_SHARING_MODE_EXCLUSIVE = 0
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x02
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x04
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x01
VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020
VK_PIPELINE_BIND_POINT_COMPUTE = 1
VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001
VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002
VK_QUEUE_COMPUTE_BIT = 0x00000002
VK_WHOLE_SIZE = ~0  # UINT64_MAX

VK_MAX_MEMORY_TYPES = 32
VK_MAX_MEMORY_HEAPS = 16

# Vulkan handle types (opaque pointers)
VkInstance = ctypes.c_void_p
VkPhysicalDevice = ctypes.c_void_p
VkDevice = ctypes.c_void_p
VkQueue = ctypes.c_void_p
VkBuffer = ctypes.c_void_p
VkDeviceMemory = ctypes.c_void_p
VkShaderModule = ctypes.c_void_p
VkDescriptorSetLayout = ctypes.c_void_p
VkPipelineLayout = ctypes.c_void_p
VkPipeline = ctypes.c_void_p
VkDescriptorPool = ctypes.c_void_p
VkDescriptorSet = ctypes.c_void_p
VkCommandPool = ctypes.c_void_p
VkCommandBuffer = ctypes.c_void_p
VkFence = ctypes.c_void_p

VkDeviceSize = ctypes.c_uint64
VkFlags = ctypes.c_uint32
VkResult = ctypes.c_int32
VkBool32 = ctypes.c_uint32


# ---------------------------------------------------------------------------
# Vulkan structures
# ---------------------------------------------------------------------------
class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


class VkDeviceQueueCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("queueFamilyIndex", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("pQueuePriorities", ctypes.POINTER(ctypes.c_float)),
    ]


class VkPhysicalDeviceFeatures(ctypes.Structure):
    _fields_ = [(f"feature{i}", VkBool32) for i in range(55)]


class VkDeviceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("queueCreateInfoCount", ctypes.c_uint32),
        ("pQueueCreateInfos", ctypes.POINTER(VkDeviceQueueCreateInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
        ("pEnabledFeatures", ctypes.POINTER(VkPhysicalDeviceFeatures)),
    ]


class VkQueueFamilyProperties(ctypes.Structure):
    _fields_ = [
        ("queueFlags", VkFlags),
        ("queueCount", ctypes.c_uint32),
        ("timestampValidBits", ctypes.c_uint32),
        ("minImageTransferGranularity_width", ctypes.c_uint32),
        ("minImageTransferGranularity_height", ctypes.c_uint32),
        ("minImageTransferGranularity_depth", ctypes.c_uint32),
    ]


class VkMemoryType(ctypes.Structure):
    _fields_ = [
        ("propertyFlags", VkFlags),
        ("heapIndex", ctypes.c_uint32),
    ]


class VkMemoryHeap(ctypes.Structure):
    _fields_ = [
        ("size", VkDeviceSize),
        ("flags", VkFlags),
    ]


class VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", ctypes.c_uint32),
        ("memoryTypes", VkMemoryType * VK_MAX_MEMORY_TYPES),
        ("memoryHeapCount", ctypes.c_uint32),
        ("memoryHeaps", VkMemoryHeap * VK_MAX_MEMORY_HEAPS),
    ]


class VkBufferCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("size", VkDeviceSize),
        ("usage", VkFlags),
        ("sharingMode", ctypes.c_uint32),
        ("queueFamilyIndexCount", ctypes.c_uint32),
        ("pQueueFamilyIndices", ctypes.c_void_p),
    ]


class VkMemoryRequirements(ctypes.Structure):
    _fields_ = [
        ("size", VkDeviceSize),
        ("alignment", VkDeviceSize),
        ("memoryTypeBits", ctypes.c_uint32),
    ]


class VkMemoryAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("allocationSize", VkDeviceSize),
        ("memoryTypeIndex", ctypes.c_uint32),
    ]


class VkShaderModuleCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("codeSize", ctypes.c_size_t),
        ("pCode", ctypes.c_void_p),
    ]


class VkDescriptorSetLayoutBinding(ctypes.Structure):
    _fields_ = [
        ("binding", ctypes.c_uint32),
        ("descriptorType", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
        ("stageFlags", VkFlags),
        ("pImmutableSamplers", ctypes.c_void_p),
    ]


class VkDescriptorSetLayoutCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("bindingCount", ctypes.c_uint32),
        ("pBindings", ctypes.POINTER(VkDescriptorSetLayoutBinding)),
    ]


class VkPushConstantRange(ctypes.Structure):
    _fields_ = [
        ("stageFlags", VkFlags),
        ("offset", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
    ]


class VkPipelineLayoutCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("setLayoutCount", ctypes.c_uint32),
        ("pSetLayouts", ctypes.POINTER(VkDescriptorSetLayout)),
        ("pushConstantRangeCount", ctypes.c_uint32),
        ("pPushConstantRanges", ctypes.POINTER(VkPushConstantRange)),
    ]


class VkSpecializationMapEntry(ctypes.Structure):
    _fields_ = [
        ("constantID", ctypes.c_uint32),
        ("offset", ctypes.c_uint32),
        ("size", ctypes.c_size_t),
    ]


class VkSpecializationInfo(ctypes.Structure):
    _fields_ = [
        ("mapEntryCount", ctypes.c_uint32),
        ("pMapEntries", ctypes.POINTER(VkSpecializationMapEntry)),
        ("dataSize", ctypes.c_size_t),
        ("pData", ctypes.c_void_p),
    ]


class VkPipelineShaderStageCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("stage", VkFlags),
        ("module", VkShaderModule),
        ("pName", ctypes.c_char_p),
        ("pSpecializationInfo", ctypes.POINTER(VkSpecializationInfo)),
    ]


class VkComputePipelineCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("stage", VkPipelineShaderStageCreateInfo),
        ("layout", VkPipelineLayout),
        ("basePipelineHandle", VkPipeline),
        ("basePipelineIndex", ctypes.c_int32),
    ]


class VkDescriptorPoolSize(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
    ]


class VkDescriptorPoolCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("maxSets", ctypes.c_uint32),
        ("poolSizeCount", ctypes.c_uint32),
        ("pPoolSizes", ctypes.POINTER(VkDescriptorPoolSize)),
    ]


class VkDescriptorSetAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("descriptorPool", VkDescriptorPool),
        ("descriptorSetCount", ctypes.c_uint32),
        ("pSetLayouts", ctypes.POINTER(VkDescriptorSetLayout)),
    ]


class VkDescriptorBufferInfo(ctypes.Structure):
    _fields_ = [
        ("buffer", VkBuffer),
        ("offset", VkDeviceSize),
        ("range", VkDeviceSize),
    ]


class VkWriteDescriptorSet(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("dstSet", VkDescriptorSet),
        ("dstBinding", ctypes.c_uint32),
        ("dstArrayElement", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
        ("descriptorType", ctypes.c_uint32),
        ("pImageInfo", ctypes.c_void_p),
        ("pBufferInfo", ctypes.POINTER(VkDescriptorBufferInfo)),
        ("pTexelBufferView", ctypes.c_void_p),
    ]


class VkCommandPoolCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("queueFamilyIndex", ctypes.c_uint32),
    ]


class VkCommandBufferAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("commandPool", VkCommandPool),
        ("level", ctypes.c_uint32),
        ("commandBufferCount", ctypes.c_uint32),
    ]


class VkCommandBufferBeginInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("pInheritanceInfo", ctypes.c_void_p),
    ]


class VkSubmitInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("waitSemaphoreCount", ctypes.c_uint32),
        ("pWaitSemaphores", ctypes.c_void_p),
        ("pWaitDstStageMask", ctypes.c_void_p),
        ("commandBufferCount", ctypes.c_uint32),
        ("pCommandBuffers", ctypes.POINTER(VkCommandBuffer)),
        ("signalSemaphoreCount", ctypes.c_uint32),
        ("pSignalSemaphores", ctypes.c_void_p),
    ]


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------
def _load_vulkan():
    """Load the Vulkan shared library."""
    system = platform.system()
    if system == "Linux":
        names = ["libvulkan.so.1", "libvulkan.so"]
    elif system == "Darwin":
        names = ["libvulkan.dylib", "libMoltenVK.dylib"]
    elif system == "Windows":
        names = ["vulkan-1.dll"]
    else:
        names = []

    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue

    # Try ctypes.util as fallback
    path = ctypes.util.find_library("vulkan")
    if path:
        return ctypes.CDLL(path)

    raise RuntimeError(
        "Could not find Vulkan library. Install the Vulkan SDK or GPU driver.")


_vk = None


def _get_vk():
    global _vk
    if _vk is None:
        _vk = _load_vulkan()
    return _vk


def _check_vk(result, msg="Vulkan"):
    """Check a VkResult, raise on error."""
    if result != VK_SUCCESS:
        raise RuntimeError(f"{msg} failed with VkResult {result}")
    return result


# ---------------------------------------------------------------------------
# VulkanBuffer
# ---------------------------------------------------------------------------
class VulkanBuffer(DeviceBuffer):
    """Host-visible coherent buffer backed by Vulkan device memory.

    Memory is persistently mapped for zero-copy access on integrated GPUs.
    """

    def __init__(self, backend, numpy_dtype, shape):
        self._backend = backend
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(np.prod(shape)) * self._numpy_dtype.itemsize
        # Ensure minimum size of 4 bytes (Vulkan requires non-zero buffer)
        alloc_size = max(self._nbytes, 4)

        vk = _get_vk()
        device = backend._device

        # Create buffer
        buf_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext=None, flags=0,
            size=alloc_size,
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0, pQueueFamilyIndices=None,
        )
        self._vk_buffer = VkBuffer()
        _check_vk(vk.vkCreateBuffer(device, ctypes.byref(buf_info), None,
                                      ctypes.byref(self._vk_buffer)),
                   "vkCreateBuffer")

        # Query memory requirements
        mem_req = VkMemoryRequirements()
        vk.vkGetBufferMemoryRequirements(device, self._vk_buffer,
                                          ctypes.byref(mem_req))

        # Find suitable memory type
        mem_type_idx = backend._find_memory_type(
            mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

        # Allocate memory
        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=None,
            allocationSize=mem_req.size,
            memoryTypeIndex=mem_type_idx,
        )
        self._vk_memory = VkDeviceMemory()
        _check_vk(vk.vkAllocateMemory(device, ctypes.byref(alloc_info), None,
                                        ctypes.byref(self._vk_memory)),
                   "vkAllocateMemory")

        # Bind memory to buffer
        _check_vk(vk.vkBindBufferMemory(device, self._vk_buffer,
                                          self._vk_memory, 0),
                   "vkBindBufferMemory")

        # Map memory persistently
        self._mapped_ptr = ctypes.c_void_p()
        _check_vk(vk.vkMapMemory(device, self._vk_memory, 0, alloc_size, 0,
                                   ctypes.byref(self._mapped_ptr)),
                   "vkMapMemory")

        # Zero-initialize
        ctypes.memset(self._mapped_ptr, 0, alloc_size)

    def from_numpy(self, arr: np.ndarray):
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        ctypes.memmove(self._mapped_ptr, src.ctypes.data, self._nbytes)

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self._shape, dtype=self._numpy_dtype)
        ctypes.memmove(out.ctypes.data, self._mapped_ptr, self._nbytes)
        return out

    def fill(self, value):
        arr = np.full(self._shape, value, dtype=self._numpy_dtype)
        self.from_numpy(arr)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def destroy(self):
        """Explicitly free Vulkan resources."""
        if not hasattr(self, '_backend') or self._backend is None:
            return
        if not self._backend._alive:
            return
        vk = _get_vk()
        device = self._backend._device
        if self._mapped_ptr:
            vk.vkUnmapMemory(device, self._vk_memory)
            self._mapped_ptr = None
        if self._vk_buffer:
            vk.vkDestroyBuffer(device, self._vk_buffer, None)
            self._vk_buffer = None
        if self._vk_memory:
            vk.vkFreeMemory(device, self._vk_memory, None)
            self._vk_memory = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CompiledVulkanKernel
# ---------------------------------------------------------------------------
class CompiledVulkanKernel:
    """A compiled Vulkan compute pipeline ready for dispatch."""

    def __init__(self, pipeline, pipeline_layout, desc_set_layout,
                 num_bindings, workgroup_size):
        self._pipeline = pipeline
        self._pipeline_layout = pipeline_layout
        self._desc_set_layout = desc_set_layout
        self._num_bindings = num_bindings
        self._workgroup_size = workgroup_size

    def __call__(self, kernel_args, param_is_field, param_types,
                 loop_end, backend):
        """Dispatch the compute kernel."""
        vk = _get_vk()
        device = backend._device

        # Build list of VulkanBuffers for all bindings (fields + scalar wrappers)
        buffers = []
        temp_buffers = []
        for arg, is_field, ptype in zip(kernel_args, param_is_field, param_types):
            if is_field:
                buffers.append(arg._buffer)
            else:
                # Scalar arg: pack into a temporary 1-element buffer
                np_dtype = _NUMPY_DTYPE[ptype]
                tmp = VulkanBuffer(backend, np_dtype, (1,))
                tmp.from_numpy(np.array([arg], dtype=np_dtype))
                buffers.append(tmp)
                temp_buffers.append(tmp)

        # Create descriptor pool
        pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=self._num_bindings,
        )
        pool_info = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext=None, flags=0,
            maxSets=1,
            poolSizeCount=1,
            pPoolSizes=ctypes.pointer(pool_size),
        )
        desc_pool = VkDescriptorPool()
        _check_vk(vk.vkCreateDescriptorPool(device, ctypes.byref(pool_info),
                                              None, ctypes.byref(desc_pool)),
                   "vkCreateDescriptorPool")

        # Allocate descriptor set
        layouts = (VkDescriptorSetLayout * 1)(self._desc_set_layout)
        alloc_info = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext=None,
            descriptorPool=desc_pool,
            descriptorSetCount=1,
            pSetLayouts=layouts,
        )
        desc_set = VkDescriptorSet()
        _check_vk(vk.vkAllocateDescriptorSets(device, ctypes.byref(alloc_info),
                                                ctypes.byref(desc_set)),
                   "vkAllocateDescriptorSets")

        # Update descriptor set with buffer bindings
        buf_infos = (VkDescriptorBufferInfo * self._num_bindings)()
        writes = (VkWriteDescriptorSet * self._num_bindings)()
        for i, buf in enumerate(buffers):
            buf_infos[i].buffer = buf._vk_buffer
            buf_infos[i].offset = 0
            buf_infos[i].range = VK_WHOLE_SIZE
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
            writes[i].pNext = None
            writes[i].dstSet = desc_set
            writes[i].dstBinding = i
            writes[i].dstArrayElement = 0
            writes[i].descriptorCount = 1
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            writes[i].pImageInfo = None
            writes[i].pBufferInfo = ctypes.pointer(buf_infos[i])
            writes[i].pTexelBufferView = None

        vk.vkUpdateDescriptorSets(device, self._num_bindings, writes, 0, None)

        # Record command buffer
        cmd_buf = backend._cmd_buffer
        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext=None,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo=None,
        )
        _check_vk(vk.vkResetCommandBuffer(cmd_buf, 0), "vkResetCommandBuffer")
        _check_vk(vk.vkBeginCommandBuffer(cmd_buf, ctypes.byref(begin_info)),
                   "vkBeginCommandBuffer")

        vk.vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                              self._pipeline)

        desc_sets = (VkDescriptorSet * 1)(desc_set)
        vk.vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    self._pipeline_layout, 0, 1, desc_sets,
                                    0, None)

        # Push constant: loop range n (uint32)
        n_val = ctypes.c_uint32(loop_end)
        vk.vkCmdPushConstants(cmd_buf, self._pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
                               ctypes.byref(n_val))

        # Dispatch
        grid_x = (loop_end + self._workgroup_size - 1) // self._workgroup_size
        vk.vkCmdDispatch(cmd_buf, grid_x, 1, 1)

        _check_vk(vk.vkEndCommandBuffer(cmd_buf), "vkEndCommandBuffer")

        # Submit and wait
        cmd_bufs = (VkCommandBuffer * 1)(cmd_buf)
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=None,
            waitSemaphoreCount=0, pWaitSemaphores=None, pWaitDstStageMask=None,
            commandBufferCount=1, pCommandBuffers=cmd_bufs,
            signalSemaphoreCount=0, pSignalSemaphores=None,
        )
        _check_vk(vk.vkQueueSubmit(backend._queue, 1,
                                     ctypes.byref(submit_info), None),
                   "vkQueueSubmit")
        _check_vk(vk.vkQueueWaitIdle(backend._queue), "vkQueueWaitIdle")

        # Cleanup descriptor pool (frees the descriptor set too)
        vk.vkDestroyDescriptorPool(device, desc_pool, None)


# ---------------------------------------------------------------------------
# VulkanBackend
# ---------------------------------------------------------------------------
def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    from pgc.runtime.cpu import _get_loop_range as cpu_get_loop_range
    return cpu_get_loop_range(ir_func, args)


class VulkanBackend:
    """Vulkan GPU compute backend — host-visible fields, SPIR-V compilation."""

    def __init__(self):
        vk = _get_vk()

        # Create instance
        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext=None,
            pApplicationName=b"PGC",
            applicationVersion=1,
            pEngineName=b"PGC",
            engineVersion=1,
            apiVersion=VK_API_VERSION_1_1,
        )
        inst_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext=None, flags=0,
            pApplicationInfo=ctypes.pointer(app_info),
            enabledLayerCount=0, ppEnabledLayerNames=None,
            enabledExtensionCount=0, ppEnabledExtensionNames=None,
        )
        self._instance = VkInstance()
        _check_vk(vk.vkCreateInstance(ctypes.byref(inst_info), None,
                                       ctypes.byref(self._instance)),
                   "vkCreateInstance")

        # Enumerate physical devices
        count = ctypes.c_uint32(0)
        _check_vk(vk.vkEnumeratePhysicalDevices(self._instance,
                                                  ctypes.byref(count), None),
                   "vkEnumeratePhysicalDevices (count)")
        if count.value == 0:
            raise RuntimeError("No Vulkan physical devices found")

        phys_devs = (VkPhysicalDevice * count.value)()
        _check_vk(vk.vkEnumeratePhysicalDevices(self._instance,
                                                  ctypes.byref(count), phys_devs),
                   "vkEnumeratePhysicalDevices")
        self._physical_device = phys_devs[0]

        # Get memory properties
        self._mem_props = VkPhysicalDeviceMemoryProperties()
        vk.vkGetPhysicalDeviceMemoryProperties(self._physical_device,
                                                ctypes.byref(self._mem_props))

        # Find compute queue family
        qf_count = ctypes.c_uint32(0)
        vk.vkGetPhysicalDeviceQueueFamilyProperties(
            self._physical_device, ctypes.byref(qf_count), None)
        qf_props = (VkQueueFamilyProperties * qf_count.value)()
        vk.vkGetPhysicalDeviceQueueFamilyProperties(
            self._physical_device, ctypes.byref(qf_count), qf_props)

        self._queue_family = None
        for i in range(qf_count.value):
            if qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT:
                self._queue_family = i
                break
        if self._queue_family is None:
            raise RuntimeError("No compute queue family found")

        # Create logical device with one compute queue
        priority = (ctypes.c_float * 1)(1.0)
        queue_info = VkDeviceQueueCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext=None, flags=0,
            queueFamilyIndex=self._queue_family,
            queueCount=1,
            pQueuePriorities=priority,
        )
        dev_info = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=None, flags=0,
            queueCreateInfoCount=1,
            pQueueCreateInfos=ctypes.pointer(queue_info),
            enabledLayerCount=0, ppEnabledLayerNames=None,
            enabledExtensionCount=0, ppEnabledExtensionNames=None,
            pEnabledFeatures=None,
        )
        self._device = VkDevice()
        _check_vk(vk.vkCreateDevice(self._physical_device,
                                      ctypes.byref(dev_info), None,
                                      ctypes.byref(self._device)),
                   "vkCreateDevice")

        # Get compute queue
        self._queue = VkQueue()
        vk.vkGetDeviceQueue(self._device, self._queue_family, 0,
                             ctypes.byref(self._queue))

        # Create command pool
        pool_info = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext=None,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self._queue_family,
        )
        self._cmd_pool = VkCommandPool()
        _check_vk(vk.vkCreateCommandPool(self._device, ctypes.byref(pool_info),
                                           None, ctypes.byref(self._cmd_pool)),
                   "vkCreateCommandPool")

        # Allocate one reusable command buffer
        alloc_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext=None,
            commandPool=self._cmd_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self._cmd_buffer = VkCommandBuffer()
        _check_vk(vk.vkAllocateCommandBuffers(self._device,
                                                ctypes.byref(alloc_info),
                                                ctypes.byref(self._cmd_buffer)),
                   "vkAllocateCommandBuffers")

        self._cache: dict[str, CompiledVulkanKernel] = {}
        self._alive = True

    def _find_memory_type(self, type_bits: int, required_flags: int) -> int:
        """Find a memory type index matching the requirements."""
        # Prefer DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (resizable BAR)
        for i in range(self._mem_props.memoryTypeCount):
            if not (type_bits & (1 << i)):
                continue
            flags = self._mem_props.memoryTypes[i].propertyFlags
            if (flags & (required_flags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) == \
               (required_flags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT):
                return i
        # Fall back to HOST_VISIBLE | HOST_COHERENT
        for i in range(self._mem_props.memoryTypeCount):
            if not (type_bits & (1 << i)):
                continue
            flags = self._mem_props.memoryTypes[i].propertyFlags
            if (flags & required_flags) == required_flags:
                return i
        raise RuntimeError("No suitable memory type found")

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...]) -> VulkanBuffer:
        return VulkanBuffer(self, dtype.numpy_dtype, shape)

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the Vulkan GPU."""
        if kwargs:
            raise NotImplementedError("Keyword arguments not supported in kernels")

        from pgc.runtime.cpu import (
            _detect_template_args, _expand_template_args,
            _detect_vector_fields_from_args,
        )
        template_args = _detect_template_args(kernel, args)
        effective_args = _expand_template_args(args, template_args)

        vector_fields = _detect_vector_fields_from_args(kernel, args, template_args)

        ir_module = kernel.get_ir(
            vector_fields,
            template_args=template_args if template_args else None,
        )
        ir_func = ir_module.functions[0]

        # Resolve dimension sizes
        name_to_field = {}
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Field):
                name_to_field[param.name] = arg
        from pgc.lang.ir_resolve import resolve_ir
        resolve_ir(ir_func, name_to_field)

        # Type inference
        infer_param_types(ir_func, effective_args)

        # Optimization passes
        from pgc.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

        # Cache key
        type_sig = tuple(p.type_annotation for p in ir_func.params)
        tmpl_key = ""
        if template_args:
            tmpl_key = str(kernel._make_cache_key(vector_fields, template_args))
        cache_key = f"{kernel.name}_{id(kernel)}_{type_sig}_{tmpl_key}"

        if cache_key not in self._cache:
            self._cache[cache_key] = self._compile_kernel(ir_func)

        compiled = self._cache[cache_key]

        kernel_args = list(effective_args)
        loop_end = _get_loop_range(ir_func, effective_args)

        param_types = [p.type_annotation for p in ir_func.params]
        param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]

        compiled(kernel_args, param_is_field, param_types, loop_end, self)

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledVulkanKernel:
        """Compile PGC IR → SPIR-V → Vulkan compute pipeline."""
        vk = _get_vk()
        workgroup_size = 256

        spirv_bytes = generate_spirv(ir_func, workgroup_size=workgroup_size)

        # Create shader module
        shader_info = VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            pNext=None, flags=0,
            codeSize=len(spirv_bytes),
            pCode=ctypes.cast(ctypes.c_char_p(spirv_bytes), ctypes.c_void_p),
        )
        shader_module = VkShaderModule()
        _check_vk(vk.vkCreateShaderModule(self._device,
                                            ctypes.byref(shader_info), None,
                                            ctypes.byref(shader_module)),
                   "vkCreateShaderModule")

        # Descriptor set layout: one storage buffer per param
        num_bindings = len(ir_func.params)
        bindings = (VkDescriptorSetLayoutBinding * num_bindings)()
        for i in range(num_bindings):
            bindings[i].binding = i
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            bindings[i].descriptorCount = 1
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            bindings[i].pImmutableSamplers = None

        layout_info = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext=None, flags=0,
            bindingCount=num_bindings,
            pBindings=bindings,
        )
        desc_set_layout = VkDescriptorSetLayout()
        _check_vk(vk.vkCreateDescriptorSetLayout(
            self._device, ctypes.byref(layout_info), None,
            ctypes.byref(desc_set_layout)),
            "vkCreateDescriptorSetLayout")

        # Pipeline layout with push constant for n
        push_range = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=4,
        )
        layouts = (VkDescriptorSetLayout * 1)(desc_set_layout)
        pipe_layout_info = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext=None, flags=0,
            setLayoutCount=1,
            pSetLayouts=layouts,
            pushConstantRangeCount=1,
            pPushConstantRanges=ctypes.pointer(push_range),
        )
        pipeline_layout = VkPipelineLayout()
        _check_vk(vk.vkCreatePipelineLayout(
            self._device, ctypes.byref(pipe_layout_info), None,
            ctypes.byref(pipeline_layout)),
            "vkCreatePipelineLayout")

        # Compute pipeline
        stage_info = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext=None, flags=0,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName=b"main",
            pSpecializationInfo=None,
        )
        pipe_info = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            pNext=None, flags=0,
            stage=stage_info,
            layout=pipeline_layout,
            basePipelineHandle=None,
            basePipelineIndex=-1,
        )
        pipeline = VkPipeline()
        _check_vk(vk.vkCreateComputePipelines(
            self._device, None, 1, ctypes.byref(pipe_info), None,
            ctypes.byref(pipeline)),
            "vkCreateComputePipelines")

        # Shader module no longer needed
        vk.vkDestroyShaderModule(self._device, shader_module, None)

        return CompiledVulkanKernel(pipeline, pipeline_layout, desc_set_layout,
                                    num_bindings, workgroup_size)

    def __del__(self):
        # Mark as dead so VulkanBuffer destructors skip cleanup
        self._alive = False
