"""PGC Vulkan backend — compiles kernels via SPIR-V and dispatches via Vulkan compute.

Pipeline:
    PGC IR → SPIR-V binary → vkCreateShaderModule → compute pipeline → vkCmdDispatch

Fields use host-visible coherent memory (zero-copy on integrated GPUs, mapped on discrete).
No per-dispatch copies — data stays on the GPU between kernel calls.

Requires libvulkan.so (Linux), libvulkan.dylib (macOS/MoltenVK), or vulkan-1.dll (Windows).
No Python Vulkan packages needed — uses ctypes directly.
"""

import atexit
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
VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR = 0x00000001

VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001
VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002
VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020
VK_SHARING_MODE_EXCLUSIVE = 0
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x02
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x04
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x01
VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1
VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020
VK_PIPELINE_BIND_POINT_COMPUTE = 1
VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001
VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002
VK_QUEUE_COMPUTE_BIT = 0x00000002
VK_WHOLE_SIZE = ~0  # UINT64_MAX

# Image / sampler constants
VK_IMAGE_TYPE_3D = 2
VK_IMAGE_VIEW_TYPE_3D = 5
VK_FORMAT_R32_SFLOAT = 100
VK_IMAGE_TILING_OPTIMAL = 0
VK_IMAGE_USAGE_TRANSFER_DST_BIT = 0x00000002
VK_IMAGE_USAGE_SAMPLED_BIT = 0x00000004
VK_IMAGE_LAYOUT_UNDEFINED = 0
VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL = 5
VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL = 7
VK_FILTER_LINEAR = 1
VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE = 2
VK_SAMPLER_MIPMAP_MODE_NEAREST = 0
VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK = 3
VK_COMPONENT_SWIZZLE_IDENTITY = 0
VK_IMAGE_ASPECT_COLOR_BIT = 0x00000001
VK_SAMPLE_COUNT_1_BIT = 0x00000001
VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001
VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000
VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800
VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000
VK_ACCESS_SHADER_READ_BIT = 0x00000020
VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO = 14
VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = 15
VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO = 31
VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER = 45
VK_QUEUE_FAMILY_IGNORED = 0xFFFFFFFF

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
VkImage = ctypes.c_void_p
VkImageView = ctypes.c_void_p
VkSampler = ctypes.c_void_p

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


class VkBufferCopy(ctypes.Structure):
    _fields_ = [
        ("srcOffset", VkDeviceSize),
        ("dstOffset", VkDeviceSize),
        ("size", VkDeviceSize),
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
# Image/sampler structures
# ---------------------------------------------------------------------------
class VkExtent3D(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
    ]


class VkImageCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("imageType", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
        ("extent", VkExtent3D),
        ("mipLevels", ctypes.c_uint32),
        ("arrayLayers", ctypes.c_uint32),
        ("samples", ctypes.c_uint32),
        ("tiling", ctypes.c_uint32),
        ("usage", VkFlags),
        ("sharingMode", ctypes.c_uint32),
        ("queueFamilyIndexCount", ctypes.c_uint32),
        ("pQueueFamilyIndices", ctypes.c_void_p),
        ("initialLayout", ctypes.c_uint32),
    ]


class VkImageSubresourceRange(ctypes.Structure):
    _fields_ = [
        ("aspectMask", VkFlags),
        ("baseMipLevel", ctypes.c_uint32),
        ("levelCount", ctypes.c_uint32),
        ("baseArrayLayer", ctypes.c_uint32),
        ("layerCount", ctypes.c_uint32),
    ]


class VkComponentMapping(ctypes.Structure):
    _fields_ = [
        ("r", ctypes.c_uint32),
        ("g", ctypes.c_uint32),
        ("b", ctypes.c_uint32),
        ("a", ctypes.c_uint32),
    ]


class VkImageViewCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("image", VkImage),
        ("viewType", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
        ("components", VkComponentMapping),
        ("subresourceRange", VkImageSubresourceRange),
    ]


class VkSamplerCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", VkFlags),
        ("magFilter", ctypes.c_uint32),
        ("minFilter", ctypes.c_uint32),
        ("mipmapMode", ctypes.c_uint32),
        ("addressModeU", ctypes.c_uint32),
        ("addressModeV", ctypes.c_uint32),
        ("addressModeW", ctypes.c_uint32),
        ("mipLodBias", ctypes.c_float),
        ("anisotropyEnable", VkBool32),
        ("maxAnisotropy", ctypes.c_float),
        ("compareEnable", VkBool32),
        ("compareOp", ctypes.c_uint32),
        ("minLod", ctypes.c_float),
        ("maxLod", ctypes.c_float),
        ("borderColor", ctypes.c_uint32),
        ("unnormalizedCoordinates", VkBool32),
    ]


class VkDescriptorImageInfo(ctypes.Structure):
    _fields_ = [
        ("sampler", VkSampler),
        ("imageView", VkImageView),
        ("imageLayout", ctypes.c_uint32),
    ]


class VkImageMemoryBarrier(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("srcAccessMask", VkFlags),
        ("dstAccessMask", VkFlags),
        ("oldLayout", ctypes.c_uint32),
        ("newLayout", ctypes.c_uint32),
        ("srcQueueFamilyIndex", ctypes.c_uint32),
        ("dstQueueFamilyIndex", ctypes.c_uint32),
        ("image", VkImage),
        ("subresourceRange", VkImageSubresourceRange),
    ]


class VkImageSubresourceLayers(ctypes.Structure):
    _fields_ = [
        ("aspectMask", VkFlags),
        ("mipLevel", ctypes.c_uint32),
        ("baseArrayLayer", ctypes.c_uint32),
        ("layerCount", ctypes.c_uint32),
    ]


class VkOffset3D(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("z", ctypes.c_int32),
    ]


class VkBufferImageCopy(ctypes.Structure):
    _fields_ = [
        ("bufferOffset", VkDeviceSize),
        ("bufferRowLength", ctypes.c_uint32),
        ("bufferImageHeight", ctypes.c_uint32),
        ("imageSubresource", VkImageSubresourceLayers),
        ("imageOffset", VkOffset3D),
        ("imageExtent", VkExtent3D),
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


def _setup_argtypes(vk):
    """Set argtypes/restype for all Vulkan functions to ensure correct 64-bit pointer handling."""
    P = ctypes.c_void_p  # pointer args
    U32 = ctypes.c_uint32
    U64 = ctypes.c_uint64

    vk.vkCreateInstance.argtypes = [P, P, P]
    vk.vkCreateInstance.restype = VkResult
    vk.vkDestroyInstance.argtypes = [P, P]
    vk.vkDestroyInstance.restype = None
    vk.vkEnumeratePhysicalDevices.argtypes = [P, P, P]
    vk.vkEnumeratePhysicalDevices.restype = VkResult
    vk.vkGetPhysicalDeviceMemoryProperties.argtypes = [P, P]
    vk.vkGetPhysicalDeviceMemoryProperties.restype = None
    vk.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [P, P, P]
    vk.vkGetPhysicalDeviceQueueFamilyProperties.restype = None
    vk.vkCreateDevice.argtypes = [P, P, P, P]
    vk.vkCreateDevice.restype = VkResult
    vk.vkDestroyDevice.argtypes = [P, P]
    vk.vkDestroyDevice.restype = None
    vk.vkDeviceWaitIdle.argtypes = [P]
    vk.vkDeviceWaitIdle.restype = VkResult
    vk.vkGetDeviceQueue.argtypes = [P, U32, U32, P]
    vk.vkGetDeviceQueue.restype = None
    vk.vkCreateBuffer.argtypes = [P, P, P, P]
    vk.vkCreateBuffer.restype = VkResult
    vk.vkDestroyBuffer.argtypes = [P, P, P]
    vk.vkDestroyBuffer.restype = None
    vk.vkGetBufferMemoryRequirements.argtypes = [P, P, P]
    vk.vkGetBufferMemoryRequirements.restype = None
    vk.vkAllocateMemory.argtypes = [P, P, P, P]
    vk.vkAllocateMemory.restype = VkResult
    vk.vkFreeMemory.argtypes = [P, P, P]
    vk.vkFreeMemory.restype = None
    vk.vkBindBufferMemory.argtypes = [P, P, P, U64]
    vk.vkBindBufferMemory.restype = VkResult
    vk.vkMapMemory.argtypes = [P, P, U64, U64, U32, P]
    vk.vkMapMemory.restype = VkResult
    vk.vkUnmapMemory.argtypes = [P, P]
    vk.vkUnmapMemory.restype = None
    vk.vkCreateShaderModule.argtypes = [P, P, P, P]
    vk.vkCreateShaderModule.restype = VkResult
    vk.vkDestroyShaderModule.argtypes = [P, P, P]
    vk.vkDestroyShaderModule.restype = None
    vk.vkCreateDescriptorSetLayout.argtypes = [P, P, P, P]
    vk.vkCreateDescriptorSetLayout.restype = VkResult
    vk.vkDestroyDescriptorSetLayout.argtypes = [P, P, P]
    vk.vkDestroyDescriptorSetLayout.restype = None
    vk.vkCreatePipelineLayout.argtypes = [P, P, P, P]
    vk.vkCreatePipelineLayout.restype = VkResult
    vk.vkDestroyPipelineLayout.argtypes = [P, P, P]
    vk.vkDestroyPipelineLayout.restype = None
    vk.vkCreateComputePipelines.argtypes = [P, P, U32, P, P, P]
    vk.vkCreateComputePipelines.restype = VkResult
    vk.vkDestroyPipeline.argtypes = [P, P, P]
    vk.vkDestroyPipeline.restype = None
    vk.vkCreateDescriptorPool.argtypes = [P, P, P, P]
    vk.vkCreateDescriptorPool.restype = VkResult
    vk.vkDestroyDescriptorPool.argtypes = [P, P, P]
    vk.vkDestroyDescriptorPool.restype = None
    vk.vkAllocateDescriptorSets.argtypes = [P, P, P]
    vk.vkAllocateDescriptorSets.restype = VkResult
    vk.vkUpdateDescriptorSets.argtypes = [P, U32, P, U32, P]
    vk.vkUpdateDescriptorSets.restype = None
    vk.vkCreateCommandPool.argtypes = [P, P, P, P]
    vk.vkCreateCommandPool.restype = VkResult
    vk.vkDestroyCommandPool.argtypes = [P, P, P]
    vk.vkDestroyCommandPool.restype = None
    vk.vkAllocateCommandBuffers.argtypes = [P, P, P]
    vk.vkAllocateCommandBuffers.restype = VkResult
    vk.vkResetCommandBuffer.argtypes = [P, U32]
    vk.vkResetCommandBuffer.restype = VkResult
    vk.vkBeginCommandBuffer.argtypes = [P, P]
    vk.vkBeginCommandBuffer.restype = VkResult
    vk.vkEndCommandBuffer.argtypes = [P]
    vk.vkEndCommandBuffer.restype = VkResult
    vk.vkCmdBindPipeline.argtypes = [P, U32, P]
    vk.vkCmdBindPipeline.restype = None
    vk.vkCmdBindDescriptorSets.argtypes = [P, U32, P, U32, U32, P, U32, P]
    vk.vkCmdBindDescriptorSets.restype = None
    vk.vkCmdPushConstants.argtypes = [P, P, U32, U32, U32, P]
    vk.vkCmdPushConstants.restype = None
    vk.vkCmdDispatch.argtypes = [P, U32, U32, U32]
    vk.vkCmdDispatch.restype = None
    vk.vkCmdCopyBuffer.argtypes = [P, P, P, U32, P]
    vk.vkCmdCopyBuffer.restype = None
    vk.vkCmdFillBuffer.argtypes = [P, P, U64, U64, U32]
    vk.vkCmdFillBuffer.restype = None
    vk.vkQueueSubmit.argtypes = [P, U32, P, P]
    vk.vkQueueSubmit.restype = VkResult
    vk.vkQueueWaitIdle.argtypes = [P]
    vk.vkQueueWaitIdle.restype = VkResult

    # Image / sampler functions
    vk.vkCreateImage.argtypes = [P, P, P, P]
    vk.vkCreateImage.restype = VkResult
    vk.vkDestroyImage.argtypes = [P, P, P]
    vk.vkDestroyImage.restype = None
    vk.vkGetImageMemoryRequirements.argtypes = [P, P, P]
    vk.vkGetImageMemoryRequirements.restype = None
    vk.vkBindImageMemory.argtypes = [P, P, P, U64]
    vk.vkBindImageMemory.restype = VkResult
    vk.vkCreateImageView.argtypes = [P, P, P, P]
    vk.vkCreateImageView.restype = VkResult
    vk.vkDestroyImageView.argtypes = [P, P, P]
    vk.vkDestroyImageView.restype = None
    vk.vkCreateSampler.argtypes = [P, P, P, P]
    vk.vkCreateSampler.restype = VkResult
    vk.vkDestroySampler.argtypes = [P, P, P]
    vk.vkDestroySampler.restype = None
    vk.vkCmdPipelineBarrier.argtypes = [P, U32, U32, U32, U32, P, U32, P, U32, P]
    vk.vkCmdPipelineBarrier.restype = None
    vk.vkCmdCopyBufferToImage.argtypes = [P, P, P, U32, U32, P]
    vk.vkCmdCopyBufferToImage.restype = None


def _get_vk():
    global _vk
    if _vk is None:
        _vk = _load_vulkan()
        _setup_argtypes(_vk)
    return _vk


def _check_vk(result, msg="Vulkan"):
    """Check a VkResult, raise on error."""
    if result != VK_SUCCESS:
        raise RuntimeError(f"{msg} failed with VkResult {result}")
    return result


# ---------------------------------------------------------------------------
# Helpers for creating Vulkan buffers and memory
# ---------------------------------------------------------------------------
def _create_bound_buffer(vk, device, size, usage, mem_type_idx):
    """Create a VkBuffer, allocate memory, bind, return (buffer, memory, actual_size)."""
    buf_info = VkBufferCreateInfo(
        sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        pNext=None, flags=0,
        size=size,
        usage=usage,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        queueFamilyIndexCount=0, pQueueFamilyIndices=None,
    )
    buf = VkBuffer()
    _check_vk(vk.vkCreateBuffer(device, ctypes.byref(buf_info), None,
                                  ctypes.byref(buf)),
               "vkCreateBuffer")

    mem_req = VkMemoryRequirements()
    vk.vkGetBufferMemoryRequirements(device, buf, ctypes.byref(mem_req))

    alloc_info = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        pNext=None,
        allocationSize=mem_req.size,
        memoryTypeIndex=mem_type_idx,
    )
    mem = VkDeviceMemory()
    _check_vk(vk.vkAllocateMemory(device, ctypes.byref(alloc_info), None,
                                    ctypes.byref(mem)),
               "vkAllocateMemory")
    _check_vk(vk.vkBindBufferMemory(device, buf, mem, 0), "vkBindBufferMemory")
    return buf, mem


def _copy_buffer(backend, src_buf, dst_buf, size):
    """Record and submit a buffer-to-buffer copy command."""
    vk = _get_vk()
    cmd = backend._transfer_cmd
    begin_info = VkCommandBufferBeginInfo(
        sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        pNext=None,
        flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        pInheritanceInfo=None,
    )
    _check_vk(vk.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer")
    _check_vk(vk.vkBeginCommandBuffer(cmd, ctypes.byref(begin_info)),
               "vkBeginCommandBuffer")
    region = VkBufferCopy(srcOffset=0, dstOffset=0, size=size)
    vk.vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, ctypes.byref(region))
    _check_vk(vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer")

    cmd_bufs = (VkCommandBuffer * 1)(cmd)
    submit = VkSubmitInfo(
        sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        pNext=None,
        waitSemaphoreCount=0, pWaitSemaphores=None, pWaitDstStageMask=None,
        commandBufferCount=1, pCommandBuffers=cmd_bufs,
        signalSemaphoreCount=0, pSignalSemaphores=None,
    )
    _check_vk(vk.vkQueueSubmit(backend._queue, 1, ctypes.byref(submit), None),
               "vkQueueSubmit")
    _check_vk(vk.vkQueueWaitIdle(backend._queue), "vkQueueWaitIdle")


# ---------------------------------------------------------------------------
# VulkanBuffer
# ---------------------------------------------------------------------------
class VulkanBuffer(DeviceBuffer):
    """Device-resident buffer backed by Vulkan device memory.

    On discrete GPUs, data lives in DEVICE_LOCAL VRAM. Transfers use a
    staging buffer in HOST_VISIBLE memory:
        from_numpy → staging → vkCmdCopyBuffer → device
        to_numpy   → vkCmdCopyBuffer → staging → host read

    On integrated GPUs (or if no device-local memory), falls back to
    HOST_VISIBLE | HOST_COHERENT with direct mapped access (zero-copy).
    """

    def __init__(self, backend, numpy_dtype, shape):
        self._backend = backend
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(np.prod(shape)) * self._numpy_dtype.itemsize
        alloc_size = max(self._nbytes, 4)

        vk = _get_vk()
        device = backend._device

        self._staging_buf = None
        self._staging_mem = None
        self._mapped_ptr = None
        self._vk_buffer = None
        self._vk_memory = None
        self._device_local = False

        if backend._has_device_local:
            # Discrete GPU: device-local buffer + host-visible staging buffer
            usage_device = (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT)
            self._vk_buffer, self._vk_memory = _create_bound_buffer(
                vk, device, alloc_size, usage_device, backend._device_local_mem_type)

            usage_staging = (VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT)
            self._staging_buf, self._staging_mem = _create_bound_buffer(
                vk, device, alloc_size, usage_staging, backend._host_visible_mem_type)

            # Map staging buffer persistently
            self._mapped_ptr = ctypes.c_void_p()
            _check_vk(vk.vkMapMemory(device, self._staging_mem, 0, alloc_size, 0,
                                       ctypes.byref(self._mapped_ptr)),
                       "vkMapMemory")
            self._device_local = True
        else:
            # Integrated GPU: host-visible buffer, direct access
            usage = (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT)
            self._vk_buffer, self._vk_memory = _create_bound_buffer(
                vk, device, alloc_size, usage, backend._host_visible_mem_type)

            self._mapped_ptr = ctypes.c_void_p()
            _check_vk(vk.vkMapMemory(device, self._vk_memory, 0, alloc_size, 0,
                                       ctypes.byref(self._mapped_ptr)),
                       "vkMapMemory")

        # Zero-initialize
        ctypes.memset(self._mapped_ptr, 0, alloc_size)
        if self._device_local:
            # Use vkCmdFillBuffer for efficient device-side zeroing
            vk2 = _get_vk()
            cmd = backend._transfer_cmd
            begin_info = VkCommandBufferBeginInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext=None,
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                pInheritanceInfo=None,
            )
            _check_vk(vk2.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer")
            _check_vk(vk2.vkBeginCommandBuffer(cmd, ctypes.byref(begin_info)),
                       "vkBeginCommandBuffer")
            vk2.vkCmdFillBuffer(cmd, self._vk_buffer, 0, alloc_size, 0)
            _check_vk(vk2.vkEndCommandBuffer(cmd), "vkEndCommandBuffer")
            cmd_bufs = (VkCommandBuffer * 1)(cmd)
            submit = VkSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, pNext=None,
                waitSemaphoreCount=0, pWaitSemaphores=None, pWaitDstStageMask=None,
                commandBufferCount=1, pCommandBuffers=cmd_bufs,
                signalSemaphoreCount=0, pSignalSemaphores=None,
            )
            _check_vk(vk2.vkQueueSubmit(backend._queue, 1, ctypes.byref(submit), None),
                       "vkQueueSubmit")
            _check_vk(vk2.vkQueueWaitIdle(backend._queue), "vkQueueWaitIdle")

    def from_numpy(self, arr: np.ndarray):
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        ctypes.memmove(self._mapped_ptr, src.ctypes.data, self._nbytes)
        if self._device_local:
            _copy_buffer(self._backend, self._staging_buf, self._vk_buffer,
                         max(self._nbytes, 4))

    def to_numpy(self) -> np.ndarray:
        if self._device_local:
            _copy_buffer(self._backend, self._vk_buffer, self._staging_buf,
                         max(self._nbytes, 4))
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
        if not self._backend._alive or not self._backend._device:
            return
        vk = _get_vk()
        device = self._backend._device
        if self._mapped_ptr:
            mem_to_unmap = self._staging_mem if self._device_local else self._vk_memory
            if mem_to_unmap:
                vk.vkUnmapMemory(device, mem_to_unmap)
            self._mapped_ptr = None
        if self._staging_buf:
            vk.vkDestroyBuffer(device, self._staging_buf, None)
            self._staging_buf = None
        if self._staging_mem:
            vk.vkFreeMemory(device, self._staging_mem, None)
            self._staging_mem = None
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
                 num_bindings, workgroup_size,
                 param_is_texture=None, texture_shapes=None):
        self._pipeline = pipeline
        self._pipeline_layout = pipeline_layout
        self._desc_set_layout = desc_set_layout
        self._num_bindings = num_bindings
        self._workgroup_size = workgroup_size
        self._param_is_texture = param_is_texture or [False] * num_bindings
        self._texture_shapes = texture_shapes or {}
        self._tex_cache: dict[tuple, tuple] = {}  # cache_key → (image, mem, view, sampler)

    def _create_texture(self, field, W, H, D, backend):
        """Create a VkImage + VkImageView + VkSampler for 3D texture sampling."""
        vk = _get_vk()
        device = backend._device

        # Create 3D image
        img_info = VkImageCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            pNext=None, flags=0,
            imageType=VK_IMAGE_TYPE_3D,
            format=VK_FORMAT_R32_SFLOAT,
            extent=VkExtent3D(width=W, height=H, depth=D),
            mipLevels=1, arrayLayers=1,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_OPTIMAL,
            usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0, pQueueFamilyIndices=None,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        )
        image = VkImage()
        _check_vk(vk.vkCreateImage(device, ctypes.byref(img_info), None,
                                    ctypes.byref(image)),
                   "vkCreateImage")

        # Allocate and bind device-local memory
        mem_req = VkMemoryRequirements()
        vk.vkGetImageMemoryRequirements(device, image, ctypes.byref(mem_req))
        mem_type = backend._find_memory_type(
            mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=None,
            allocationSize=mem_req.size,
            memoryTypeIndex=mem_type,
        )
        mem = VkDeviceMemory()
        _check_vk(vk.vkAllocateMemory(device, ctypes.byref(alloc_info), None,
                                       ctypes.byref(mem)),
                   "vkAllocateMemory (image)")
        _check_vk(vk.vkBindImageMemory(device, image, mem, 0), "vkBindImageMemory")

        # Transition layout: UNDEFINED → TRANSFER_DST_OPTIMAL
        # Copy buffer → image
        # Transition layout: TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        subres_range = VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1,
            baseArrayLayer=0, layerCount=1,
        )

        cmd = backend._transfer_cmd
        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext=None,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo=None,
        )
        _check_vk(vk.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer")
        _check_vk(vk.vkBeginCommandBuffer(cmd, ctypes.byref(begin_info)),
                   "vkBeginCommandBuffer")

        # Barrier: UNDEFINED → TRANSFER_DST
        barrier1 = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            pNext=None,
            srcAccessMask=0,
            dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
            oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=subres_range,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, None,
            0, None,
            1, ctypes.byref(barrier1))

        # Copy buffer → image
        region = VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=VkImageSubresourceLayers(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1),
            imageOffset=VkOffset3D(x=0, y=0, z=0),
            imageExtent=VkExtent3D(width=W, height=H, depth=D),
        )
        # Use the device buffer as source — it has TRANSFER_SRC_BIT.
        src_buf = field._buffer._vk_buffer
        vk.vkCmdCopyBufferToImage(
            cmd, src_buf, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, ctypes.byref(region))

        # Barrier: TRANSFER_DST → SHADER_READ_ONLY
        barrier2 = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            pNext=None,
            srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
            oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=subres_range,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, None,
            0, None,
            1, ctypes.byref(barrier2))

        _check_vk(vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer")

        cmd_bufs = (VkCommandBuffer * 1)(cmd)
        submit = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=None,
            waitSemaphoreCount=0, pWaitSemaphores=None, pWaitDstStageMask=None,
            commandBufferCount=1, pCommandBuffers=cmd_bufs,
            signalSemaphoreCount=0, pSignalSemaphores=None,
        )
        _check_vk(vk.vkQueueSubmit(backend._queue, 1, ctypes.byref(submit), None),
                   "vkQueueSubmit")
        _check_vk(vk.vkQueueWaitIdle(backend._queue), "vkQueueWaitIdle")

        # Create image view
        view_info = VkImageViewCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            pNext=None, flags=0,
            image=image,
            viewType=VK_IMAGE_VIEW_TYPE_3D,
            format=VK_FORMAT_R32_SFLOAT,
            components=VkComponentMapping(
                r=VK_COMPONENT_SWIZZLE_IDENTITY,
                g=VK_COMPONENT_SWIZZLE_IDENTITY,
                b=VK_COMPONENT_SWIZZLE_IDENTITY,
                a=VK_COMPONENT_SWIZZLE_IDENTITY),
            subresourceRange=subres_range,
        )
        view = VkImageView()
        _check_vk(vk.vkCreateImageView(device, ctypes.byref(view_info), None,
                                        ctypes.byref(view)),
                   "vkCreateImageView")

        # Create sampler (linear filtering, normalized coords, clamp)
        sampler_info = VkSamplerCreateInfo(
            sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            pNext=None, flags=0,
            magFilter=VK_FILTER_LINEAR,
            minFilter=VK_FILTER_LINEAR,
            mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST,
            addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            mipLodBias=0.0,
            anisotropyEnable=0,
            maxAnisotropy=1.0,
            compareEnable=0,
            compareOp=0,
            minLod=0.0,
            maxLod=0.0,
            borderColor=VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
            unnormalizedCoordinates=0,
        )
        sampler = VkSampler()
        _check_vk(vk.vkCreateSampler(device, ctypes.byref(sampler_info), None,
                                      ctypes.byref(sampler)),
                   "vkCreateSampler")

        return image, mem, view, sampler

    def __call__(self, kernel_args, param_is_field, param_types,
                 loop_end, backend):
        """Dispatch the compute kernel."""
        vk = _get_vk()
        device = backend._device

        # Build list of VulkanBuffers (for storage buffers) and texture info
        buffers = []       # index → VulkanBuffer or None (for textures)
        tex_infos = {}     # index → (image, mem, view, sampler)
        temp_buffers = []
        for i, (arg, is_field, ptype, is_tex) in enumerate(
                zip(kernel_args, param_is_field, param_types,
                    self._param_is_texture)):
            if is_tex:
                W, H, D = self._texture_shapes[i]
                cache_key = (id(arg._buffer._vk_buffer), W, H, D)
                if cache_key not in self._tex_cache:
                    self._tex_cache[cache_key] = self._create_texture(
                        arg, W, H, D, backend)
                tex_infos[i] = self._tex_cache[cache_key]
                buffers.append(None)
            elif is_field:
                buffers.append(arg._buffer)
            else:
                np_dtype = _NUMPY_DTYPE[ptype]
                tmp = VulkanBuffer(backend, np_dtype, (1,))
                tmp.from_numpy(np.array([arg], dtype=np_dtype))
                buffers.append(tmp)
                temp_buffers.append(tmp)

        # Count descriptor types for pool
        n_storage = sum(1 for t in self._param_is_texture if not t)
        n_sampler = sum(1 for t in self._param_is_texture if t)

        pool_sizes = []
        if n_storage > 0:
            pool_sizes.append(VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=n_storage))
        if n_sampler > 0:
            pool_sizes.append(VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=n_sampler))
        pool_sizes_arr = (VkDescriptorPoolSize * len(pool_sizes))(*pool_sizes)

        pool_info = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext=None, flags=0,
            maxSets=1,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes_arr,
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

        # Update descriptor set — mixed buffer and image bindings
        # Keep arrays alive until vkUpdateDescriptorSets returns
        buf_infos = (VkDescriptorBufferInfo * self._num_bindings)()
        img_infos = (VkDescriptorImageInfo * self._num_bindings)()
        writes = (VkWriteDescriptorSet * self._num_bindings)()
        for i in range(self._num_bindings):
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
            writes[i].pNext = None
            writes[i].dstSet = desc_set
            writes[i].dstBinding = i
            writes[i].dstArrayElement = 0
            writes[i].descriptorCount = 1
            writes[i].pTexelBufferView = None

            if self._param_is_texture[i]:
                _, _, view, sampler = tex_infos[i]
                img_infos[i].sampler = sampler
                img_infos[i].imageView = view
                img_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                writes[i].pImageInfo = ctypes.cast(
                    ctypes.pointer(img_infos[i]), ctypes.c_void_p)
                writes[i].pBufferInfo = None
            else:
                buf_infos[i].buffer = buffers[i]._vk_buffer
                buf_infos[i].offset = 0
                buf_infos[i].range = VK_WHOLE_SIZE
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                writes[i].pImageInfo = None
                writes[i].pBufferInfo = ctypes.pointer(buf_infos[i])

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
def _optimize_spirv(spirv_bytes: bytes) -> bytes:
    """Run spirv-opt on the SPIR-V binary if available.

    Performs SSA promotion, dead code elimination, and other optimizations
    that dramatically reduce instruction count (typically 60-70% reduction).
    Falls back to unoptimized SPIR-V if spirv-opt is not installed.
    """
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("spirv-opt"):
        return spirv_bytes

    try:
        with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as f_in:
            f_in.write(spirv_bytes)
            f_in.flush()
            out_path = f_in.name + ".opt"
            result = subprocess.run(
                ["spirv-opt", "-O", f_in.name, "-o", out_path],
                capture_output=True, timeout=10)
            if result.returncode == 0:
                with open(out_path, "rb") as f_out:
                    optimized = f_out.read()
                import os
                os.unlink(f_in.name)
                os.unlink(out_path)
                return optimized
            import os
            os.unlink(f_in.name)
    except Exception:
        pass
    return spirv_bytes


def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    from pgc.runtime.cpu import _get_loop_range as cpu_get_loop_range
    return cpu_get_loop_range(ir_func, args)


def _build_reduce_kernels():
    """Build PGC reduction kernels using a chunked parallel approach.

    Each of P threads reduces a contiguous chunk of ~n/P elements
    sequentially, writing its partial result to a partials buffer.
    The host then reduces P partial results via numpy.

    This avoids shared memory barriers (which conflict with PGC's
    bounds-guard early-return) and atomic contention.
    """
    import pgc as _pgc

    @_pgc.kernel
    def _partial_sum(x, partials, chunk_size, total_n):
        for tid in range(partials.shape[0]):
            start = tid * int(chunk_size)
            end = start + int(chunk_size)
            if end > int(total_n):
                end = int(total_n)
            acc = 0.0
            j = start
            while j < end:
                acc = acc + x[j]
                j = j + 1
            partials[tid] = acc

    @_pgc.kernel
    def _partial_min(x, partials, chunk_size, total_n):
        for tid in range(partials.shape[0]):
            start = tid * int(chunk_size)
            end = start + int(chunk_size)
            if end > int(total_n):
                end = int(total_n)
            acc = x[start]
            j = start + 1
            while j < end:
                acc = min(acc, x[j])
                j = j + 1
            partials[tid] = acc

    @_pgc.kernel
    def _partial_max(x, partials, chunk_size, total_n):
        for tid in range(partials.shape[0]):
            start = tid * int(chunk_size)
            end = start + int(chunk_size)
            if end > int(total_n):
                end = int(total_n)
            acc = x[start]
            j = start + 1
            while j < end:
                acc = max(acc, x[j])
                j = j + 1
            partials[tid] = acc

    return {"sum": _partial_sum, "min": _partial_min, "max": _partial_max}


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
        # On macOS (MoltenVK), we need the portability enumeration extension
        import sys
        inst_flags = 0
        inst_extensions = []
        if sys.platform == "darwin":
            inst_flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
            inst_extensions.append(b"VK_KHR_portability_enumeration")

        if inst_extensions:
            ext_array = (ctypes.c_char_p * len(inst_extensions))(*inst_extensions)
            ext_ptr = ctypes.cast(ext_array, ctypes.c_void_p)
            ext_count = len(inst_extensions)
        else:
            ext_ptr = None
            ext_count = 0

        inst_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext=None, flags=inst_flags,
            pApplicationInfo=ctypes.pointer(app_info),
            enabledLayerCount=0, ppEnabledLayerNames=None,
            enabledExtensionCount=ext_count, ppEnabledExtensionNames=ext_ptr,
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

        # Allocate a second command buffer for transfer operations
        alloc_info2 = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext=None,
            commandPool=self._cmd_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self._transfer_cmd = VkCommandBuffer()
        _check_vk(vk.vkAllocateCommandBuffers(self._device,
                                                ctypes.byref(alloc_info2),
                                                ctypes.byref(self._transfer_cmd)),
                   "vkAllocateCommandBuffers (transfer)")

        # Detect memory types for device-local and host-visible allocation
        self._device_local_mem_type = None
        self._host_visible_mem_type = None
        self._has_device_local = False

        host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        for i in range(self._mem_props.memoryTypeCount):
            flags = self._mem_props.memoryTypes[i].propertyFlags
            # Pure device-local (NOT host-visible) — VRAM on discrete GPUs
            if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT and
                    not (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)):
                if self._device_local_mem_type is None:
                    self._device_local_mem_type = i
            # Host-visible coherent (NOT device-local) — system RAM
            if (flags & host_flags) == host_flags:
                if not (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT):
                    if self._host_visible_mem_type is None:
                        self._host_visible_mem_type = i

        # Fallback: any host-visible coherent type
        if self._host_visible_mem_type is None:
            for i in range(self._mem_props.memoryTypeCount):
                flags = self._mem_props.memoryTypes[i].propertyFlags
                if (flags & host_flags) == host_flags:
                    self._host_visible_mem_type = i
                    break

        self._has_device_local = (self._device_local_mem_type is not None and
                                   self._host_visible_mem_type is not None)

        self._cache: dict[str, CompiledVulkanKernel] = {}
        self._alive = True
        atexit.register(self._cleanup)

    def _find_memory_type(self, type_bits: int, required_flags: int,
                          prefer_device_local: bool = True) -> int:
        """Find a memory type index matching the requirements."""
        if prefer_device_local:
            # Prefer DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (resizable BAR)
            for i in range(self._mem_props.memoryTypeCount):
                if not (type_bits & (1 << i)):
                    continue
                flags = self._mem_props.memoryTypes[i].propertyFlags
                if (flags & (required_flags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) == \
                   (required_flags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT):
                    return i
        # Fall back to HOST_VISIBLE | HOST_COHERENT (system memory)
        for i in range(self._mem_props.memoryTypeCount):
            if not (type_bits & (1 << i)):
                continue
            flags = self._mem_props.memoryTypes[i].propertyFlags
            if (flags & required_flags) == required_flags:
                return i
        raise RuntimeError("No suitable memory type found")

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...]) -> VulkanBuffer:
        return VulkanBuffer(self, dtype.numpy_dtype, shape)

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an external Vulkan buffer. Not yet implemented."""
        raise NotImplementedError(
            "Vulkan pointer interop is not yet supported. "
            "Use allocate_field + from_numpy instead.")

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the Vulkan GPU."""
        if kwargs:
            raise NotImplementedError("Keyword arguments not supported in kernels")

        from pgc.runtime.cpu import (
            _detect_template_args, _expand_template_args,
            _detect_vector_fields_from_args, _detect_texture_fields,
        )
        from pgc.lang.field import Texture3D
        template_args = _detect_template_args(kernel, args)
        effective_args = _expand_template_args(args, template_args)

        vector_fields = _detect_vector_fields_from_args(kernel, args, template_args)
        texture_fields = _detect_texture_fields(kernel, args, template_args)

        ir_module = kernel.get_ir(
            vector_fields,
            template_args=template_args if template_args else None,
            texture_fields=texture_fields,
        )
        ir_func = ir_module.functions[0]

        # Resolve dimension sizes and texture shapes
        name_to_field = {}
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                name_to_field[param.name] = arg
            elif isinstance(arg, Field):
                name_to_field[param.name] = arg
        from pgc.lang.ir_resolve import resolve_ir
        resolve_ir(ir_func, name_to_field)

        # Type inference
        infer_param_types(ir_func, effective_args)

        # Store texture shapes on params for codegen/dispatch
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                param._texture_shape = arg.shape_3d

        # Optimization passes
        from pgc.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

        # Determine loop range BEFORE packing
        kernel_args = [a.field if isinstance(a, Texture3D) else a
                       for a in effective_args]
        loop_end = _get_loop_range(ir_func, kernel_args)

        # Cache key (include texture shapes for uniqueness)
        type_sig = tuple(p.type_annotation for p in ir_func.params)
        tex_sig = tuple(
            getattr(p, '_texture_shape', None) for p in ir_func.params)
        tmpl_key = ""
        if template_args:
            tmpl_key = str(kernel._make_cache_key(vector_fields, template_args))
        cache_key = f"{kernel.name}_{id(kernel)}_{type_sig}_{tex_sig}_{tmpl_key}"

        if cache_key not in self._cache:
            import copy
            from pgc.lang.ir_pack_scalars import pack_scalars
            from pgc.lang.ir_type_annotate import annotate_types
            from pgc.runtime.cpu import _create_pack_fields
            ir_func_copy = copy.deepcopy(ir_func)
            _, pack_info = pack_scalars(ir_func_copy, effective_args)
            annotate_types(ir_func_copy)
            compiled = self._compile_kernel(ir_func_copy)
            packed_param_types = [p.type_annotation for p in ir_func_copy.params]
            packed_param_is_field = [getattr(p, '_is_field', True) for p in ir_func_copy.params]
            pack_fields = _create_pack_fields(pack_info, effective_args, self) if pack_info else None
            self._cache[cache_key] = (compiled, pack_info, packed_param_types, packed_param_is_field, pack_fields)

        compiled, pack_info, param_types, param_is_field, pack_fields = self._cache[cache_key]

        # Build dispatch args
        if pack_info:
            from pgc.runtime.cpu import _update_pack_fields
            from pgc.lang.ir_pack_scalars import split_args
            _update_pack_fields(pack_fields, pack_info, effective_args)
            kept_args = split_args(effective_args, pack_info)
            kernel_args = [a.field if isinstance(a, Texture3D) else a
                           for a in kept_args]
            kernel_args = list(kernel_args) + pack_fields
        else:
            kernel_args = [a.field if isinstance(a, Texture3D) else a
                           for a in effective_args]

        compiled(kernel_args, param_is_field, param_types, loop_end, self)

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledVulkanKernel:
        """Compile PGC IR → SPIR-V → Vulkan compute pipeline."""
        vk = _get_vk()
        workgroup_size = 256

        spirv_bytes = generate_spirv(ir_func, workgroup_size=workgroup_size)
        spirv_bytes = _optimize_spirv(spirv_bytes)

        # Debug: save SPIR-V for analysis
        import os
        if os.environ.get("PGC_DUMP_SPIRV"):
            path = f"/tmp/pgc_{ir_func.name}.spv"
            with open(path, "wb") as _f:
                _f.write(spirv_bytes)
            print(f"[PGC] SPIR-V saved: {path} ({len(spirv_bytes)} bytes, {len(ir_func.params)} params)")

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

        # Extract texture metadata
        param_is_texture = [getattr(p, '_is_texture', False) for p in ir_func.params]
        texture_shapes = {}
        for i, p in enumerate(ir_func.params):
            if getattr(p, '_is_texture', False) and hasattr(p, '_texture_shape'):
                texture_shapes[i] = p._texture_shape

        # Descriptor set layout: storage buffer or combined image/sampler per param
        num_bindings = len(ir_func.params)
        bindings = (VkDescriptorSetLayoutBinding * num_bindings)()
        for i in range(num_bindings):
            bindings[i].binding = i
            bindings[i].descriptorType = (VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                          if param_is_texture[i]
                                          else VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
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
                                    num_bindings, workgroup_size,
                                    param_is_texture, texture_shapes)

    def reduce_field(self, field, op: str) -> float:
        """GPU-side reduction: sum, min, or max.

        Uses a chunked parallel approach: P GPU threads each reduce ~n/P
        elements sequentially, then the host reduces P partial results.
        """
        from pgc.lang.types import f32
        if field.dtype is not f32:
            return float(getattr(field.to_numpy(), op)())

        if not hasattr(self, '_reduce_kernels'):
            self._reduce_kernels = _build_reduce_kernels()

        kern = self._reduce_kernels[op]
        n = int(np.prod(field.shape))

        # Use enough threads for good parallelism, each handling a chunk
        num_threads = min(n, 4096)
        chunk_size = (n + num_threads - 1) // num_threads

        import pgc as _pgc
        partials = _pgc.field(dtype=_pgc.f32, shape=(num_threads,))

        kern(field, partials, float(chunk_size), float(n))

        partial_np = partials.to_numpy()
        return float(getattr(partial_np, op)())

    def _cleanup(self):
        """Orderly shutdown — called via atexit before Python tears down objects."""
        if not getattr(self, '_alive', False):
            return
        self._alive = False
        try:
            vk = _get_vk()
            vk.vkDeviceWaitIdle(self._device)
            for compiled in self._cache.values():
                vk.vkDestroyPipeline(self._device, compiled._pipeline, None)
                vk.vkDestroyPipelineLayout(self._device,
                                            compiled._pipeline_layout, None)
                vk.vkDestroyDescriptorSetLayout(self._device,
                                                 compiled._desc_set_layout, None)
            self._cache.clear()
            if self._cmd_pool:
                vk.vkDestroyCommandPool(self._device, self._cmd_pool, None)
                self._cmd_pool = None
            vk.vkDestroyDevice(self._device, None)
            self._device = None
            vk.vkDestroyInstance(self._instance, None)
            self._instance = None
        except Exception:
            pass

    def __del__(self):
        self._cleanup()
