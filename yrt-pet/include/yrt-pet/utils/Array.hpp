/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/utils/Concurrency.hpp"
#include "yrt-pet/utils/Globals.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace yrt
{

void py_setup_array(py::module& m);

}
#endif

namespace yrt
{

// Version tag for array files (first int in file)
constexpr int MAGIC_NUMBER = 732174000;

/** Array classes
 *
 * Classes to manage multidimensional arrays.  The memory can either be owned
 * (e.g. Array1D) or aliased (e.g Array1DAlias).
 *
 * Notes
 *
 * - When a vector of ints is used as an input, the contiguous dimension is last
 *   (e.g. in 3D indices are z/y/x).
 *
 * - Square bracket accessors can be used to get/set the values in the array,
 *   but for performance considerations, extracting flat pointers and using
 *   pointer arithmetic may be beneficial in inner loops.
 *
 * - Array classes are designed to interface transparently with numpy arrays via
 *   pybind11.
 */


template <int ndim, typename T>
class Array
{
public:
	static constexpr int NDim = ndim;
	using DType = T;

	Array() : _data(nullptr) { _shape.fill(0ull); }

	virtual ~Array() = default;

	bool isMemoryValid() const { return getRawPointer() != nullptr; }

	size_t getFlatIdx(const std::array<size_t, ndim>& idx) const
	{
		size_t flat_idx = 0;
		size_t stride = 1;
		for (int dim = ndim - 1; dim >= 0; --dim)
		{
			flat_idx += idx[dim] * stride;
			stride *= _shape[dim];
		}
		return flat_idx;
	}

	std::array<size_t, ndim> unravelIdx(size_t flatIdx) const
	{
		size_t flatIdxRemains = flatIdx;
		std::array<size_t, ndim> strides = getStridesIdx();
		std::array<size_t, ndim> indices;
		for (size_t dim = 0; dim < ndim; ++dim)
		{
			indices[dim] = flatIdxRemains / strides[dim];
			flatIdxRemains -= indices[dim] * strides[dim];
		}
		return indices;
	}

	T& get(const std::array<size_t, ndim>& idx) const
	{
		return _data[getFlatIdx(idx)];
	}

	void set(const std::array<size_t, ndim>& l, T val)
	{
		_data[getFlatIdx(l)] = val;
	}

	void increment(const std::array<size_t, ndim>& idx, T val)
	{
		_data[getFlatIdx(idx)] += val;
	}

	void incrementFlat(size_t idx, T val) { _data[idx] += val; }

	void scale(const std::array<size_t, ndim>& idx, T val)
	{
		_data[getFlatIdx(idx)] *= val;
	}

	void setFlat(size_t idx, T val) { _data[idx] = val; }
	T& getFlat(size_t idx) const { return _data[idx]; }

	size_t getSize(size_t dim) const
	{
		if (!isMemoryValid())
		{
			return 0;
		}
		if (dim >= ndim)
		{
			return 1;
		}
		return _shape[dim];
	}

	size_t getSizeTotal() const
	{
		if (!isMemoryValid())
		{
			return 0;
		}
		return totalSizeFromShape(_shape);
	}

	static size_t totalSizeFromShape(const std::array<size_t, ndim>& shape)
	{
		size_t size = 1;
		for (int dim = 0; dim < ndim; dim++)
		{
			size *= shape[dim];
		}
		return size;
	}

	std::array<size_t, ndim> getDims() const { return _shape; }

	std::array<size_t, ndim> getStrides() const
	{
		return getStridesInternal<sizeof(T)>();
	}

	std::array<size_t, ndim> getStridesIdx() const
	{
		return getStridesInternal<1ull>();
	}

	void fill(T val) { std::fill(_data, _data + getSizeTotal(), val); }

	void writeToFile(const std::string& fname) const
	{
		std::ofstream file;
		file.open(fname.c_str(), std::ios::binary | std::ios::out);
		if (!file.is_open())
		{
			throw std::filesystem::filesystem_error(
			    "The file given \"" + fname + "\" could not be opened",
			    std::make_error_code(std::errc::io_error));
		}
		int magic = MAGIC_NUMBER;
		int num_dims = ndim;
		const size_t* shape_ptr = &_shape[0];
		file.write((char*)&magic, sizeof(int));
		file.write((char*)&num_dims, sizeof(int));
		file.write((char*)shape_ptr, ndim * sizeof(size_t));
		file.write((char*)_data, getSizeTotal() * sizeof(T));
	}

	void readFromFile(const std::string& fname)
	{
		std::array<size_t, ndim> expected_dims;
		std::fill(expected_dims.begin(), expected_dims.end(), 0);
		readFromFile(fname, expected_dims);
	}

	void readFromFile(const std::string& fname,
	                  const std::array<size_t, ndim>& expected_dims)
	{
		int num_dims = 0;
		int magic = 0;
		std::ifstream file;
		file.open(fname.c_str(), std::ios::binary | std::ios::in);
		if (!file.is_open())
		{
			throw std::filesystem::filesystem_error(
			    "The file given \"" + fname + "\" could not be opened",
			    std::make_error_code(std::errc::no_such_file_or_directory));
		}

		// Get the file size
		file.seekg(0, std::ios::end);
		const size_t fileSize = file.tellg();
		file.seekg(0, std::ios::beg);

		file.read(reinterpret_cast<char*>(&magic), sizeof(int));
		if (magic != MAGIC_NUMBER)
		{
			throw std::runtime_error("The file given \"" + fname +
			                         "\" does not have a proper MAGIC_NUMBER");
		}
		file.read(reinterpret_cast<char*>(&num_dims), sizeof(int));
		if (num_dims != ndim)
		{
			throw std::runtime_error(
			    "The file given \"" + fname +
			    "\" does not have the correct number of "
			    "dimensions. Namely, the file claims to have " +
			    std::to_string(num_dims) +
			    " dimensions instead of the expected " + std::to_string(ndim) +
			    " dimensions");
		}

		auto dims = std::array<size_t, ndim>();
		file.read(reinterpret_cast<char*>(dims.data()), ndim * sizeof(size_t));

		if (expected_dims.size())
		{
			if (expected_dims.size() != (size_t)num_dims)
			{
				throw std::runtime_error(
				    "The file given \"" + fname +
				    "\" does not have the correct number of "
				    "dimensions. Namely, the file has " +
				    std::to_string(num_dims) +
				    " dimensions instead of the expected " +
				    std::to_string(expected_dims.size()) + " dimensions");
			}
			bool dim_check = true;
			for (int i = 0; i < num_dims; i++)
			{
				const size_t expected_dim = expected_dims[i];
				if (expected_dim != 0)
				{
					dim_check &= expected_dim == dims[i];
				}
			}
			if (!dim_check)
			{
				throw std::runtime_error("The file given \"" + fname +
				                         "\" has dimension sizes that do not "
				                         "match the expected sizes");
			}
		}
		const size_t totalSize = totalSizeFromShape(dims);
		constexpr size_t headerSize =
		    sizeof(int) + sizeof(int) + ndim * sizeof(size_t);
		const size_t expectedFileSize = headerSize + totalSize * sizeof(T);
		if (fileSize != expectedFileSize)
		{
			throw std::runtime_error("The file given is of the wrong size. The "
			                         "expected file size is " +
			                         std::to_string(expectedFileSize) +
			                         " while the file is " +
			                         std::to_string(fileSize) + ".");
		}

		setShape(dims);
		allocateFlat(totalSize);
		file.read(reinterpret_cast<char*>(_data), totalSize * sizeof(T));
	}

	T* getRawPointer() const { return _data; }

	Array<ndim, T>& operator+=(const Array<ndim, T>& other)
	{
		util::parallelForChunked(getSizeTotal(), globals::getNumThreads(),
		                         [this, &other](size_t i, unsigned int /*tid*/)
		                         { _data[i] += other._data[i]; });
		return *this;
	}

	Array<ndim, T>& operator+=(T other)
	{
		util::parallelForChunked(getSizeTotal(), globals::getNumThreads(),
		                         [this, other](size_t i, unsigned int /*tid*/)
		                         { _data[i] += other; });
		return *this;
	}

	Array<ndim, T>& operator*=(const Array<ndim, T>& other)
	{
		util::parallelForChunked(getSizeTotal(), globals::getNumThreads(),
		                         [this, &other](size_t i, unsigned int /*tid*/)
		                         { _data[i] *= other._data[i]; });
		return *this;
	}

	Array<ndim, T>& operator/=(const Array<ndim, T>& other)
	{
		util::parallelForChunked(getSizeTotal(), globals::getNumThreads(),
		                         [this, &other](size_t i, unsigned int /*tid*/)
		                         { _data[i] /= other._data[i]; });
		return *this;
	}

	Array<ndim, T>& operator*=(T other)
	{
		util::parallelForChunked(getSizeTotal(), globals::getNumThreads(),
		                         [this, &other](size_t i, unsigned int /*tid*/)
		                         { _data[i] *= other; });
		return *this;
	}

	Array<ndim, T>& operator/=(T other)
	{
		util::parallelForChunked(getSizeTotal(), globals::getNumThreads(),
		                         [this, other](size_t i, unsigned int /*tid*/)
		                         { _data[i] /= other; });
		return *this;
	}

	Array<ndim, T>& operator-=(const Array<ndim, T>& other)
	{
		util::parallelForChunked(getSizeTotal(), globals::getNumThreads(),
		                         [this, &other](size_t i, unsigned int /*tid*/)
		                         { _data[i] -= other._data[i]; });
		return *this;
	}

	Array<ndim, T>& operator-=(T other)
	{
		util::parallelForChunked(getSizeTotal(), globals::getNumThreads(),
		                         [this, other](size_t i, unsigned int /*tid*/)
		                         { _data[i] -= other; });
		return *this;
	}

	Array<ndim, T>& invert()
	{
		util::parallelForChunked(getSizeTotal(), globals::getNumThreads(),
		                         [this](size_t i, unsigned int /*tid*/)
		                         { _data[i] = 1 / _data[i]; });
		return *this;
	}

	T getMaxValue() const
	{
		const size_t totalSize = getSizeTotal();
		const T* arr = getRawPointer();
		std::function<T(T, T)> func_max = [](T a, T b) -> T
		{ return std::max(a, b); };
		return util::simpleReduceArray(arr, totalSize, func_max,
		                               std::numeric_limits<T>::lowest(),
		                               globals::getNumThreads());
	}

	T sum() const
	{
		const size_t totalSize = getSizeTotal();
		const T* arr = getRawPointer();
		std::function<T(T, T)> func_sum = [](T a, T b) -> T { return a + b; };
		return util::simpleReduceArray(arr, totalSize, func_sum,
		                               static_cast<T>(0),
		                               globals::getNumThreads());
	}

	// Copy from array object (memory must be allocated and appropriately sized)
	void copy(const Array<ndim, T>& src)
	{
		size_t num_el = src.getSizeTotal();
		if (num_el != getSizeTotal())
		{
			throw std::runtime_error("The source to copy has a size that does "
			                         "not match the initial array's size");
		}
		if (_data == nullptr)
		{
			throw std::runtime_error("The array has not yet been allocated, "
			                         "impossible to copy data");
		}
		std::array size = src.getDims();
		setShape(size);
		const T* data_ptr = src.getRawPointer();
		std::copy(data_ptr, data_ptr + num_el, _data);
	}

#if BUILD_PYBIND11
	// This function, although public, is only meant to be used within Python
	//  from an ArrayNDAlias.
	void bindFromNumpy(py::buffer np_data)
	{
		py::buffer_info buffer = np_data.request();

		if (buffer.ndim != NDim)
		{
			throw std::invalid_argument(
			    "The buffer given has the wrong number of dimensions");
		}

		if (buffer.format != py::format_descriptor<DType>::format())
		{
			throw std::invalid_argument(
			    "The buffer given has the wrong data type");
		}

		std::array<size_t, NDim> shape;
		for (int i = 0; i < buffer.ndim; i++)
		{
			shape[i] = buffer.shape[i];
		}

		setShape(shape);
		_data = reinterpret_cast<T*>(buffer.ptr);
	}
#endif

protected:
	std::array<size_t, ndim> _shape;
	T* _data;

	void setShape(const std::array<size_t, ndim>& shape) { _shape = shape; }

	virtual void allocateFlat(size_t size) = 0;

	std::unique_ptr<T[]> allocateFlatPointer(size_t size)
	{
		std::unique_ptr<T[]> data_ptr = nullptr;
		try
		{
			data_ptr = std::make_unique<T[]>(size);
		}
		catch (const std::bad_alloc& memoryException)
		{
			std::cerr << "Not enough memory for " << (size >> 20) << " Mb."
			          << std::endl;
			throw;
		}
		return data_ptr;
	}

	bool isShapeSet() const
	{
		for (size_t dim = 0; dim < ndim; dim++)
		{
			if (_shape[dim] == 0)
			{
				return false;
			}
		}
		return true;
	}

	template <size_t ElemSize>
	std::array<size_t, ndim> getStridesInternal() const
	{
		std::array<size_t, ndim> strides;
		for (int dim = ndim - 1; dim >= 0; --dim)
		{
			float stride;
			if (dim == ndim - 1)
			{
				stride = ElemSize;
			}
			else
			{
				stride = getSize(dim + 1) * strides[dim + 1];
			}
			strides[dim] = stride;
		}
		return strides;
	}
};


// ---------------

template <typename T>
class Array1DBase : public Array<1, T>
{
public:
	using BaseClass = Array1DBase<T>;

	Array1DBase() : Array<1, T>() {}

	T& operator[](size_t ri) const { return this->_data[ri]; }
	T& operator[](size_t ri) { return this->_data[ri]; }
};

template <typename T>
class Array1DOwned : public Array1DBase<T>
{
public:
	static constexpr bool IsOwned = true;

	Array1DOwned() : Array1DBase<T>() {}

	void allocate(size_t num_el)
	{
		if (num_el != this->getSizeTotal())
		{
			if (_data_ptr != nullptr)
			{
				_data_ptr.reset();
			}
			allocateFlat(num_el);
		}
		std::array dims = {num_el};
		this->setShape(dims);
	}

protected:
	std::unique_ptr<T[]> _data_ptr;

	void allocateFlat(size_t size) override
	{
		_data_ptr = this->allocateFlatPointer(size);
		this->_data = _data_ptr.get();
		if (_data_ptr == nullptr)
			throw std::runtime_error("Error occurred during memory allocation");
	}
};

template <typename T>
class Array1DAlias : public Array1DBase<T>
{
public:
	static constexpr bool IsOwned = false;

	Array1DAlias() : Array1DBase<T>() {}

	explicit Array1DAlias(const Array1DBase<T>& array) : Array1DBase<T>()
	{
		bind(array);
	}

	void bind(const Array1DBase<T>& array)
	{
		std::array dims = array.getDims();
		this->setShape(dims);
		this->_data = array.getRawPointer();
	}

	void bind(T* data, size_t num_el)
	{
		this->setShape({num_el});
		this->_data = data;
	}

protected:
	void allocateFlat(size_t /*size*/) override
	{
		throw std::runtime_error(
		    "Unsupported operation, cannot Allocate on Alias array");
	}
};


// ---------------


template <typename T>
class Array2DBase : public Array<2, T>
{
public:
	using BaseClass = Array2DBase<T>;

	Array2DBase() : Array<2, T>() {}

	T* operator[](size_t ri) const
	{
		return &this->_data[ri * this->_shape[1]];
	}

	T* operator[](size_t ri) { return &this->_data[ri * this->_shape[1]]; }
};

template <typename T>
class Array2DOwned : public Array2DBase<T>
{
public:
	static constexpr bool IsOwned = true;

	Array2DOwned() : Array2DBase<T>() {}

	void allocate(size_t num_rows, size_t num_el)
	{
		if (num_rows * num_el != this->getSizeTotal())
		{
			if (_data_ptr != nullptr)
			{
				_data_ptr.reset();
			}
			allocateFlat(num_rows * num_el);
		}
		this->setShape({num_rows, num_el});
	}

protected:
	std::unique_ptr<T[]> _data_ptr;

	void allocateFlat(size_t size) override
	{
		_data_ptr = this->allocateFlatPointer(size);
		this->_data = _data_ptr.get();
		if (_data_ptr == nullptr)
			throw std::runtime_error("Error occurred during memory allocation");
	}
};

template <typename T>
class Array2DAlias : public Array2DBase<T>
{
public:
	static constexpr bool IsOwned = false;

	Array2DAlias() : Array2DBase<T>() {}

	explicit Array2DAlias(const Array2DBase<T>& array) : Array2DBase<T>()
	{
		bind(array);
	}

	void bind(const Array2DBase<T>& array)
	{
		std::array dims = array.getDims();
		this->setShape(dims);
		this->_data = array.getRawPointer();
	}

	void bind(T* data, size_t num_rows, size_t num_el)
	{
		this->setShape({num_rows, num_el});
		this->_data = data;
	}

protected:
	void allocateFlat(size_t /*size*/) override
	{
		throw std::runtime_error(
		    "Unsupported operation, cannot Allocate on Alias array");
	}
};


template <typename T>
class Array3DBase : public Array<3, T>
{
public:
	using BaseClass = Array3DBase<T>;

	Array3DBase() : Array<3, T>() {}

	T* getSlicePtr(size_t ri)
	{
		return &this->_data[ri * this->_shape[1] * this->_shape[2]];
	}

	T* getSlicePtr(size_t ri) const
	{
		return &this->_data[ri * this->_shape[1] * this->_shape[2]];
	}

	Array2DAlias<T> operator[](size_t ri)
	{
		Array2DAlias<T> slice_array;
		T* data_slice = getSlicePtr(ri);
		slice_array.bind(data_slice, this->_shape[1], this->_shape[2]);
		return slice_array;
	}

	Array2DAlias<T> operator[](size_t ri) const
	{
		Array2DAlias<T> slice_array;
		T* data_slice = getSlicePtr(ri);
		slice_array.bind(data_slice, this->_shape[1], this->_shape[2]);
		return slice_array;
	}
};

template <typename T>
class Array3DOwned : public Array3DBase<T>
{
public:
	static constexpr bool IsOwned = true;

	Array3DOwned() : Array3DBase<T>() {}

	void allocate(size_t num_slices, size_t num_rows, size_t num_el)
	{
		if (num_slices * num_rows * num_el != this->getSizeTotal())
		{
			if (_data_ptr != nullptr)
			{
				_data_ptr.reset();
			}
			allocateFlat(num_slices * num_rows * num_el);
		}
		this->setShape({num_slices, num_rows, num_el});
	}

protected:
	std::unique_ptr<T[]> _data_ptr;

	void allocateFlat(size_t size) override
	{
		_data_ptr = this->allocateFlatPointer(size);
		this->_data = _data_ptr.get();
		if (_data_ptr == nullptr)
			throw std::runtime_error("Error occurred during memory allocation");
	}
};


template <typename T>
class Array3DAlias : public Array3DBase<T>
{
public:
	static constexpr bool IsOwned = false;

	Array3DAlias() : Array3DBase<T>() {}

	explicit Array3DAlias(const Array3DBase<T>& array) : Array3DBase<T>()
	{
		bind(array);
	}

	void bind(const Array3DBase<T>& array)
	{
		std::array dims = array.getDims();
		this->setShape(dims);
		this->_data = array.getRawPointer();
	}

	void bind(T* data, size_t num_slices, size_t num_rows, size_t num_el)
	{
		this->setShape({num_slices, num_rows, num_el});
		this->_data = data;
	}

protected:
	void allocateFlat(size_t /*size*/) override
	{
		throw std::runtime_error(
		    "Unsupported operation, cannot Allocate on Alias array");
	}
};


template <typename T>
class Array4DBase : public Array<4, T>
{
public:
	using BaseClass = Array4DBase<T>;

	Array4DBase() : Array<4, T>() {}

	T* getSlicePtr(size_t ri)
	{
		return &this->_data[ri * this->_shape[1] * this->_shape[2] *
		                    this->_shape[3]];
	}

	T* getSlicePtr(size_t ri) const
	{
		return &this->_data[ri * this->_shape[1] * this->_shape[2] *
		                    this->_shape[3]];
	}

	Array3DAlias<T> operator[](size_t ri)
	{
		Array3DAlias<T> slice_array;
		T* data_slice = getSlicePtr(ri);
		slice_array.bind(data_slice, this->_shape[1], this->_shape[2],
		                 this->_shape[3]);
		return slice_array;
	}

	Array3DAlias<T> operator[](size_t ri) const
	{
		Array3DAlias<T> slice_array;
		T* data_slice = getSlicePtr(ri);
		slice_array.bind(data_slice, this->_shape[1], this->_shape[2],
		                 this->_shape[3]);
		return slice_array;
	}
};

template <typename T>
class Array4DOwned : public Array4DBase<T>
{
public:
	static constexpr bool IsOwned = true;

	Array4DOwned() : Array4DBase<T>() {}

	void allocate(size_t num_t, size_t num_slices, size_t num_rows,
	              size_t num_el)
	{
		if (num_t * num_slices * num_rows * num_el != this->getSizeTotal())
		{
			if (_data_ptr != nullptr)
			{
				_data_ptr.reset();
			}
			allocateFlat(num_t * num_slices * num_rows * num_el);
		}
		this->setShape({num_t, num_slices, num_rows, num_el});
	}

protected:
	std::unique_ptr<T[]> _data_ptr;

	void allocateFlat(size_t size) override
	{
		_data_ptr = this->allocateFlatPointer(size);
		this->_data = _data_ptr.get();
		if (_data_ptr == nullptr)
			throw std::runtime_error("Error occurred during memory allocation");
	}
};


template <typename T>
class Array4DAlias : public Array4DBase<T>
{
public:
	static constexpr bool IsOwned = false;

	Array4DAlias() : Array4DBase<T>() {}

	explicit Array4DAlias(const Array4DBase<T>& array) : Array4DBase<T>()
	{
		bind(array);
	}

	void bind(const Array4DBase<T>& array)
	{
		std::array dims = array.getDims();
		this->setShape(dims);
		this->_data = array.getRawPointer();
	}

	void bind(T* data, size_t num_t, size_t num_slices, size_t num_rows,
	          size_t num_el)
	{
		this->setShape({num_t, num_slices, num_rows, num_el});
		this->_data = data;
	}

protected:
	void allocateFlat(size_t /*size*/) override
	{
		throw std::runtime_error(
		    "Unsupported operation, cannot Allocate on Alias array");
	}
};


template <typename T>
class Array5DBase : public Array<5, T>
{
public:
	using BaseClass = Array5DBase<T>;

	Array5DBase() : Array<5, T>() {}
};

template <typename T>
class Array5DOwned : public Array5DBase<T>
{
public:
	static constexpr bool IsOwned = true;

	Array5DOwned() : Array5DBase<T>() {}

	void allocate(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
	              size_t dim4)
	{
		const size_t totalSize = dim0 * dim1 * dim2 * dim3 * dim4;
		if (totalSize != this->getSizeTotal())
		{
			if (_data_ptr != nullptr)
			{
				_data_ptr.reset();
			}
			allocateFlat(totalSize);
		}
		this->setShape({dim0, dim1, dim2, dim3, dim4});
	}

protected:
	std::unique_ptr<T[]> _data_ptr;

	void allocateFlat(size_t size) override
	{
		_data_ptr = this->allocateFlatPointer(size);
		this->_data = _data_ptr.get();
		if (_data_ptr == nullptr)
			throw std::runtime_error("Error occured during memory allocation");
	}
};

template <typename T>
class Array5DAlias : public Array5DBase<T>
{
public:
	static constexpr bool IsOwned = false;

	Array5DAlias() : Array5DBase<T>() {}

	explicit Array5DAlias(const Array5DBase<T>& array) : Array5DBase<T>()
	{
		bind(array);
	}

	void bind(const Array5DBase<T>& array)
	{
		std::array dims = array.getDims();
		this->setShape(dims);
		this->_data = array.getRawPointer();
	}

	void bind(T* data, size_t dim0, size_t dim1, size_t dim2, size_t dim3,
	          size_t dim4)
	{
		this->setShape({dim0, dim1, dim2, dim3, dim4});
		this->_data = data;
	}

protected:
	void allocateFlat(size_t /*size*/) override
	{
		throw std::runtime_error(
		    "Unsupported operation, cannot Allocate on Alias array");
	}
};

}  // namespace yrt
