#ifndef TRT_DEFORM_CONV_V3_HPP
#define TRT_DEFORM_CONV_V3_HPP

#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>
#include <type_traits>
#include <stdexcept>
#include <cstdlib>

#include "NvInferRuntime.h"
#include "NvInferVersion.h"

#if NV_TENSORRT_MAJOR > 7
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    default:
      throw std::runtime_error("Invalid DataType.");
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

// Enumerator for status
typedef enum {
  STATUS_SUCCESS = 0,
  STATUS_FAILURE = 1,
  STATUS_BAD_PARAM = 2,
  STATUS_NOT_SUPPORTED = 3,
  STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

inline size_t getAlignedSize(size_t origin_size, size_t aligned_number = 16) {
  return size_t((origin_size + aligned_number - 1) / aligned_number) * aligned_number;
}

namespace nvinfer1
{
namespace plugin
{
class TRTPluginBase : public nvinfer1::IPluginV2DynamicExt {
 public:
  TRTPluginBase(const std::string &name) : mLayerName(name) {}
  // IPluginV2 Methods
  const char *getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  int initialize() TRT_NOEXCEPT override { return STATUS_SUCCESS; }
  void terminate() TRT_NOEXCEPT override {}
  void destroy() TRT_NOEXCEPT override { delete this; }
  void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override {
    mNamespace = pluginNamespace;
  }
  const char *getPluginNamespace() const TRT_NOEXCEPT override { return mNamespace.c_str(); }

  virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                               const nvinfer1::DynamicPluginTensorDesc *out,
                               int nbOutputs) TRT_NOEXCEPT override {}

  virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                  const nvinfer1::PluginTensorDesc *outputs,
                                  int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  virtual void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                               nvinfer1::IGpuAllocator *gpuAllocator) TRT_NOEXCEPT override {}

  virtual void detachFromContext() TRT_NOEXCEPT override {}

 protected:
  const std::string mLayerName;
  std::string mNamespace;

#if NV_TENSORRT_MAJOR < 8
 protected:
  // To prevent compiler warnings.
  using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::enqueue;
  using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::supportsFormat;
#endif
};

class TRTPluginCreatorBase : public nvinfer1::IPluginCreator {
 public:
  const char *getPluginVersion() const TRT_NOEXCEPT override { return "1"; };

  const nvinfer1::PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return &mFC; }

  void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override {
    mNamespace = pluginNamespace;
  }

  const char *getPluginNamespace() const TRT_NOEXCEPT override { return mNamespace.c_str(); }

 protected:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

class TRTDCNv3 : public TRTPluginBase {
 public:
  TRTDCNv3(const std::string &name, int kernel_h, int kernel_w, int stride_h, int stride_w,
           int pad_h, int pad_w, int dilation_h, int dilation_w, int group, int group_channels,
           float offset_scale, int im2col_step);

  TRTDCNv3(const std::string name, const void *data, size_t length);

  TRTDCNv3() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                          int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
      TRT_NOEXCEPT override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out,
                       int nbOutputs) TRT_NOEXCEPT override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const TRT_NOEXCEPT override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
              void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const TRT_NOEXCEPT override;

  // IPluginV2 Methods
  const char *getPluginType() const TRT_NOEXCEPT override;
  const char *getPluginVersion() const TRT_NOEXCEPT override;
  int getNbOutputs() const TRT_NOEXCEPT override;
  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void *buffer) const TRT_NOEXCEPT override;

 private:
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int dilation_h_;
  int dilation_w_;
  int group_;
  int group_channels_;
  float offset_scale_;
  int im2col_step_;
};

class TRTDCNv3Creator : public TRTPluginCreatorBase {
 public:
  TRTDCNv3Creator();

  const char *getPluginName() const TRT_NOEXCEPT override;

  const char *getPluginVersion() const TRT_NOEXCEPT override;
  nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      TRT_NOEXCEPT override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                         size_t serialLength) TRT_NOEXCEPT override;
};
}
}

template <typename T>
inline void serialize_value(void** buffer, T const& value);

template <typename T>
inline void deserialize_value(void const** buffer, size_t* buffer_size, T* value);

namespace {

template <typename T, class Enable = void>
struct Serializer {};

template <typename T>
struct Serializer<T,
                  typename std::enable_if<std::is_arithmetic<T>::value || std::is_enum<T>::value ||
                                          std::is_pod<T>::value>::type> {
  static size_t serialized_size(T const& value) { return sizeof(T); }
  static void serialize(void** buffer, T const& value) {
    ::memcpy(*buffer, &value, sizeof(T));
    reinterpret_cast<char*&>(*buffer) += sizeof(T);
  }
  static void deserialize(void const** buffer, size_t* buffer_size, T* value) {
    assert(*buffer_size >= sizeof(T));
    ::memcpy(value, *buffer, sizeof(T));
    reinterpret_cast<char const*&>(*buffer) += sizeof(T);
    *buffer_size -= sizeof(T);
  }
};

template <>
struct Serializer<const char*> {
  static size_t serialized_size(const char* value) { return strlen(value) + 1; }
  static void serialize(void** buffer, const char* value) {
    ::strcpy(static_cast<char*>(*buffer), value);
    reinterpret_cast<char*&>(*buffer) += strlen(value) + 1;
  }
  static void deserialize(void const** buffer, size_t* buffer_size, const char** value) {
    *value = static_cast<char const*>(*buffer);
    size_t data_size = strnlen(*value, *buffer_size) + 1;
    assert(*buffer_size >= data_size);
    reinterpret_cast<char const*&>(*buffer) += data_size;
    *buffer_size -= data_size;
  }
};

template <typename T>
struct Serializer<std::vector<T>,
                  typename std::enable_if<std::is_arithmetic<T>::value || std::is_enum<T>::value ||
                                          std::is_pod<T>::value>::type> {
  static size_t serialized_size(std::vector<T> const& value) {
    return sizeof(value.size()) + value.size() * sizeof(T);
  }
  static void serialize(void** buffer, std::vector<T> const& value) {
    serialize_value(buffer, value.size());
    size_t nbyte = value.size() * sizeof(T);
    ::memcpy(*buffer, value.data(), nbyte);
    reinterpret_cast<char*&>(*buffer) += nbyte;
  }
  static void deserialize(void const** buffer, size_t* buffer_size, std::vector<T>* value) {
    size_t size;
    deserialize_value(buffer, buffer_size, &size);
    value->resize(size);
    size_t nbyte = value->size() * sizeof(T);
    assert(*buffer_size >= nbyte);
    ::memcpy(value->data(), *buffer, nbyte);
    reinterpret_cast<char const*&>(*buffer) += nbyte;
    *buffer_size -= nbyte;
  }
};

}  // namespace

template <typename T>
inline size_t serialized_size(T const& value) {
  return Serializer<T>::serialized_size(value);
}

template <typename T>
inline void serialize_value(void** buffer, T const& value) {
  return Serializer<T>::serialize(buffer, value);
}

template <typename T>
inline void deserialize_value(void const** buffer, size_t* buffer_size, T* value) {
  return Serializer<T>::deserialize(buffer, buffer_size, value);
}

#endif  // TRT_DEFORM_CONV_V3_HPP
