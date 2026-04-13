/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/GPUUtils.cuh"
#include "yrt-pet/utils/Types.hpp"

#if BUILD_CUDA
#include "yrt-pet/utils/DeviceArray.cuh"
#include "yrt-pet/utils/DeviceSynchronizedObject.cuh"
#endif


#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace yrt
{

// List of properties that can be gathered from projection-space data. This,
//  however, should not list the projection value itself, as it is not meant to
//  be held by the projection properties structure
enum class ProjectionPropertyType
{
	DET_ID = 0,
	TIMESTAMP,
	LOR,
	DET_ORIENT,
	MEASUREMENT,
	TOF,
	DYNAMIC_FRAME,
	SENSITIVITY,
	ATTENUATION,
	ATTENUATION_PRECORRECTION,
	SCATTER_ESTIMATE,
	RANDOMS_ESTIMATE,
	COUNT
};

enum class ConstraintVariableType
{
	DET1 = 0,
	DET2,
	ABS_DELTA_ANGLE_DEG,
	ABS_DELTA_ANGLE_IDX,
	ABS_DELTA_BLOCK_IDX,
	COUNT
};

using PropertyUnit = uint8_t;  // One byte
static_assert(sizeof(PropertyUnit) == 1);

template <typename Enum>
class PropStructManager
{
public:
	explicit PropStructManager(const std::set<Enum>& props);

	// Helper functions
	std::unique_ptr<PropertyUnit[]> createDataArray(size_t numElements) const;
	template <typename T>
	HOST_DEVICE_CALLABLE inline T* getDataPtr(PropertyUnit* data, size_t idx,
	                                          Enum prop) const
	{
		return reinterpret_cast<T*>(data + elementSize * idx +
		                            offsetMap[static_cast<int>(prop)]);
	}

	template <typename T>
	HOST_DEVICE_CALLABLE inline const T* getDataPtr(const PropertyUnit* data,
	                                                size_t idx, Enum prop) const
	{
		return reinterpret_cast<const T*>(data + elementSize * idx +
		                                  offsetMap[static_cast<int>(prop)]);
	}

	template <typename T>
	HOST_DEVICE_CALLABLE inline void setDataValue(PropertyUnit* data,
	                                              size_t idx, Enum prop,
	                                              const T& value) const
	{
		T* ptr = getDataPtr<T>(data, idx, prop);
		*ptr = value;
	}

	template <typename T>
	HOST_DEVICE_CALLABLE inline const T&
	    getDataValue(const PropertyUnit* data, size_t idx, Enum prop) const
	{
		const T* ptr = getDataPtr<T>(data, idx, prop);
		return *ptr;
	}

	// Accessors
	unsigned int getElementSize() const;
	int getTypeID() const;
	int getOffset(Enum prop) const;
	HOST_DEVICE_CALLABLE bool has(Enum prop) const
	{
		return type & (1 << static_cast<int>(prop));
	}

	// Offset table
	std::map<Enum, std::pair<std::string, int>> getInfo() const;

	template <typename... Args>
	std::ostringstream print(PropertyUnit* data, size_t elStart = 0,
	                         size_t elEnd = 1) const
	{
		auto info = getInfo();
		std::ostringstream oss;
		oss << *this << std::endl;
		int idx = 0;
		while (offsetMap[idx] < 0)
		{
			idx++;
		}
		oss << "idx=" << idx << " ";
		for (size_t el = elStart; el < elEnd; el++)
		{
			// The fold expression will handle the loop at compile time
			int current_idx = idx;  // Start with the correct index

			// This lambda captures the state and provides a compile-time "loop"
			auto print_arg = [&, info](auto type_arg)
			{
				// Check if idx is valid before printing
				if (current_idx < static_cast<int>(Enum::COUNT))
				{
					Enum var = static_cast<Enum>(current_idx);
					oss << info.at(var).first << ":"
					    << getDataValue<decltype(type_arg)>(data, el, var);
					current_idx++;
					while (current_idx < static_cast<int>(Enum::COUNT) &&
					       offsetMap[current_idx] < 0)
					{
						current_idx++;
					}
				}
			};

			// Fold expression to apply the lambda to each type in the pack
			// We use a dummy argument to call the lambda with each type
			((print_arg(Args{}), oss << ", "), ...);

			oss << std::endl;
		}
		return oss;
	}

private:
	// Data information
	// ----------------

	// Bit mask with flag for each allowed variable
	int type;
	// Total element size
	unsigned int elementSize;
	// Offset in raw pointer for each included prop
	int offsetMap[static_cast<size_t>(Enum::COUNT)];
};

using ProjectionPropertyManager = PropStructManager<ProjectionPropertyType>;

template <typename Enum>
std::ostream& operator<<(std::ostream& oss, const PropStructManager<Enum>& t)
{
	const auto& info = t.getInfo();
	oss << "[";
	for (int i = 0; i < static_cast<int>(Enum::COUNT); i++)
	{
		if (t.getTypeID() & (1 << i))
		{
			auto var = static_cast<Enum>(i);
			oss << info.at(var).first << ": " << info.at(var).second
			    << " (offset: " << t.getOffset(var) << "), ";
		}
	}
	oss << "]";
	return oss;
}

// Wrapper to hold host-side projection-space data
template <typename Enum>
class PropStruct
{
public:
	explicit PropStruct(const std::set<Enum>& props)
	    : m_manager(props), mp_data(nullptr), m_size(0)
	{
	}

	void allocate(size_t numElements)
	{
		mp_data = m_manager.createDataArray(numElements);
		m_size = numElements;
	}

	template <typename T>
	const T* getDataPtr(size_t idx, Enum prop) const
	{
		return m_manager.template getDataPtr<T>(mp_data.get(), idx, prop);
	}

	template <typename T>
	T getValue(size_t idx, Enum prop) const
	{
		return *this->getDataPtr<T>(idx, prop);
	}

	template <typename T>
	void setValue(size_t idx, Enum prop, const T& value) const
	{
		m_manager.template setDataValue<T>(mp_data.get(), idx, prop, value);
	}

	size_t getElementSize() const { return m_manager.getElementSize(); }

	// In number of elements (aka number of rows)
	size_t size() const { return m_size; }

	const PropStructManager<Enum>* getManager() const { return &m_manager; }

	PropertyUnit* getRawPointer() const { return mp_data.get(); }
	PropertyUnit* getRawPointer() { return mp_data.get(); }

private:
	PropStructManager<Enum> m_manager;
	std::unique_ptr<PropertyUnit[]> mp_data;
	size_t m_size;  // Number of elements
};

#if BUILD_CUDA
template <typename Enum>
class PropStructDevice
{
public:
	// This object is meant to be passed (by copy) to kernels
	struct PropStructDeviceObject
	{
		PropStructManager<Enum>* mpd_manager;  // Manager on the device
		PropertyUnit* mpd_data;  // Raw pointer to the data on the device
		size_t numElements;      // Number of rows in the structure

		template <typename T>
		HOST_DEVICE_CALLABLE inline const T* getDataPtr(size_t idx,
		                                                Enum prop) const
		{
			return mpd_manager->template getDataPtr<T>(mpd_data, idx, prop);
		}

		HOST_DEVICE_CALLABLE bool has(Enum prop) const
		{
			return mpd_manager->has(prop);
		}

		template <typename T>
		HOST_DEVICE_CALLABLE inline void setValue(size_t idx, Enum prop,
		                                          const T& value) const
		{
			mpd_manager->template setDataValue<T>(mpd_data, idx, prop, value);
		}

		template <typename T>
		HOST_DEVICE_CALLABLE inline const T& getValue(size_t idx,
		                                              Enum prop) const
		{
			return mpd_manager->template getDataValue<T>(mpd_data, idx, prop);
		}
	};

	explicit PropStructDevice(const std::set<Enum>& props) : m_manager(props) {}

	bool allocate(size_t numElements, GPULaunchConfig launchConfig)
	{
		return mpd_data.allocate(numElements * getElementSize(), launchConfig);
	}

	size_t getElementSize() const
	{
		return m_manager.getHostObject().getElementSize();
	}

	bool isMemoryValid() const { return mpd_data.isMemoryValid(); }

	PropStructDeviceObject getDeviceObject() const
	{
		return {m_manager.getDevicePointer(), mpd_data.getDevicePointer(),
		        mpd_data.getSize()};
	}

	const PropStructManager<Enum>* getManager() const
	{
		return &(m_manager.getHostObject());
	}

	const PropStructManager<Enum>* getManagerDevicePointer() const
	{
		return m_manager.getDevicePointer();
	}

	// In number of elements (aka number of rows)
	size_t size() const { return mpd_data.getSize() / getElementSize(); }

	DeviceArray<PropertyUnit>& getDeviceArray() { return mpd_data; }
	PropertyUnit* getDevicePointer() { return mpd_data.getDevicePointer(); }

	void copyFromHost(const PropStruct<Enum>& hostPropStruct,
	                  GPULaunchConfig launchConfig, size_t hostNumElements)
	{
		ASSERT_MSG(
		    hostNumElements <= size(),
		    "Host-side PropStruct is larger than device-side PropStruct");
		size_t elementSize = getElementSize();
		ASSERT_MSG(elementSize == hostPropStruct.getElementSize(),
		           "Element size mismatch between host-side and device-side "
		           "prop struct");

		const size_t hostSize_bytes = hostNumElements * elementSize;

		mpd_data.copyFromHost(hostPropStruct.getRawPointer(), hostSize_bytes,
		                      launchConfig);
	}

	void copyFromHost(const PropStruct<Enum>& hostPropStruct,
	                  GPULaunchConfig launchConfig)
	{
		const size_t hostNumElements = hostPropStruct.size();
		copyFromHost(hostPropStruct, launchConfig, hostNumElements);
	}

private:
	// The manager, which is created on the host and then copied to the device
	DeviceSynchronizedObject<PropStructManager<Enum>> m_manager;
	// The raw data on the device
	DeviceArray<PropertyUnit> mpd_data;
};
#endif

}  // namespace yrt
