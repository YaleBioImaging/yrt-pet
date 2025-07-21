/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#if BUILD_PYBIND11
#include <pybind11/pybind11.h>

namespace yrt
{

void py_setup_array(pybind11::module& m);

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
	Array() : _shape(nullptr), _data(nullptr) {}

	virtual ~Array() = default;

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
		if (_shape == nullptr)
		{
			return 0;
		}
		else if (dim >= ndim)
		{
			return 1;
		}
		else
		{
			return _shape.get()[dim];
		}
	}

	size_t getSizeTotal() const
	{
		if (_shape == nullptr)
		{
			return 0;
		}
		else
		{
			size_t size = 1;
			for (int dim = 0; dim < ndim; dim++)
			{
				size *= _shape.get()[dim];
			}
			return size;
		}
	}

	void getDims(size_t output[]) const
	{
		for (int dim = 0; dim < ndim; dim++)
		{
			output[dim] = _shape.get()[dim];
		}
	}

	std::array<size_t, ndim> getDims() const
	{
		std::array<size_t, ndim> dims;
		for (int dim = 0; dim < ndim; dim++)
		{
			dims[dim] = _shape.get()[dim];
		}
		return dims;
	}

	std::array<size_t, ndim> getStrides() const
	{
		std::array<size_t, ndim> strides;
		for (int dim = ndim - 1; dim >= 0; --dim)
		{
			float stride;
			if (dim == ndim - 1)
			{
				stride = sizeof(T);
			}
			else
			{
				stride = getSize(dim + 1) * strides[dim + 1];
			}
			strides[dim] = stride;
		}
		return strides;
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
		file.write((char*)&magic, sizeof(int));
		file.write((char*)&num_dims, sizeof(int));
		file.write((char*)_shape.get(), ndim * sizeof(size_t));
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

		file.read((char*)&magic, sizeof(int));
		if (magic != MAGIC_NUMBER)
		{
			throw std::runtime_error("The file given \"" + fname +
			                         "\" does not have a proper MAGIC_NUMBER");
		}
		file.read((char*)&num_dims, sizeof(int));
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
		auto dims = std::make_unique<size_t[]>(num_dims);
		file.read((char*)dims.get(), ndim * sizeof(size_t));
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
				size_t expected_dim = expected_dims[i];
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
		setShape(dims.get());
		const size_t totalSize = getSizeTotal();
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

		allocateFlat(totalSize);
		file.read((char*)_data, totalSize * sizeof(T));
	}

	T* getRawPointer() { return _data; }
	const T* getRawPointer() const { return _data; }

	Array<ndim, T>& operator+=(const Array<ndim, T>& other)
	{
		for (size_t i = 0; i < getSizeTotal(); i++)
		{
			_data[i] += other._data[i];
		}
		return *this;
	}

	Array<ndim, T>& operator+=(T other)
	{
		for (size_t i = 0; i < getSizeTotal(); i++)
		{
			_data[i] += other;
		}
		return *this;
	}

	Array<ndim, T>& operator*=(const Array<ndim, T>& other)
	{
		for (size_t i = 0; i < getSizeTotal(); i++)
		{
			_data[i] *= other._data[i];
		}
		return *this;
	}

	Array<ndim, T>& operator/=(const Array<ndim, T>& other)
	{
		for (size_t i = 0; i < getSizeTotal(); i++)
		{
			_data[i] /= other._data[i];
		}
		return *this;
	}

	Array<ndim, T>& operator*=(T other)
	{
		for (size_t i = 0; i < getSizeTotal(); i++)
		{
			_data[i] *= other;
		}
		return *this;
	}

	Array<ndim, T>& operator/=(T other)
	{
		for (size_t i = 0; i < getSizeTotal(); i++)
		{
			_data[i] /= other;
		}
		return *this;
	}

	Array<ndim, T>& operator-=(const Array<ndim, T>& other)
	{
		for (size_t i = 0; i < getSizeTotal(); i++)
		{
			_data[i] -= other._data[i];
		}
		return *this;
	}

	Array<ndim, T>& operator-=(T other)
	{
		for (size_t i = 0; i < getSizeTotal(); i++)
		{
			_data[i] -= other;
		}
		return *this;
	}

	Array<ndim, T>& invert()
	{
		for (size_t i = 0; i < getSizeTotal(); i++)
		{
			_data[i] = 1 / _data[i];
		}
		return *this;
	}

	T getMaxValue() const
	{
		T maxValue = std::numeric_limits<T>::lowest();

		const int totalSize = getSizeTotal();
		const T* arr = getRawPointer();

#pragma omp parallel for reduction(max : maxValue) default(none) \
    firstprivate(arr, totalSize)
		for (int i = 0; i < totalSize; i++)
		{
			const float val = arr[i];
			if (val > maxValue)
			{
				maxValue = val;
			}
		}
		return maxValue;
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
		size_t size[ndim];
		src.getDims(size);
		setShape(size);
		const T* data_ptr = src.getRawPointer();
		std::copy(data_ptr, data_ptr + num_el, _data);
	}

protected:
	std::unique_ptr<size_t[]> _shape;
	T* _data;

	void setShape(size_t* dims)
	{
		if (_shape == nullptr)
		{
			_shape = std::make_unique<size_t[]>(ndim);
		}
		for (int dim = 0; dim < ndim; dim++)
		{
			_shape.get()[dim] = dims[dim];
		}
		if (_shape == nullptr)
		{
			throw std::runtime_error(
			    "Error occured while trying to change the array shape");
		}
	}

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
};


// ---------------

template <typename T>
class Array1DBase : public Array<1, T>
{
public:
	Array1DBase() : Array<1, T>()
	{
		size_t dims[1] = {0};
		this->setShape(dims);
	}

	T& operator[](size_t ri) const { return this->_data[ri]; }
	T& operator[](size_t ri) { return this->_data[ri]; }

private:
	Array1DBase(const Array1DBase<T>&) = delete;
};

template <typename T>
class Array1D : public Array1DBase<T>
{
public:
	Array1D() : Array1DBase<T>() {}

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
		size_t dims[1] = {num_el};
		this->setShape(dims);
	}

private:
	Array1D(const Array1D<T>&) = delete;

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
class Array1DAlias : public Array1DBase<T>
{
public:
	Array1DAlias() : Array1DBase<T>() {}

	void bind(const Array1DBase<T>& array)
	{
		size_t dims[1];
		array.getDims(dims);
		this->setShape(dims);
		this->_data = &(array[0]);
	}

	void bind(T* data, size_t num_el)
	{
		size_t dims[1] = {num_el};
		this->setShape(dims);
		this->_data = data;
	}

	Array1DAlias(const Array1DAlias<T>& array) : Array1DBase<T>()
	{
		bind(array);
	}

	Array1DAlias(const Array1DBase<T>* array) { bind(*array); }
	Array1DAlias(const Array1D<T>& array) { bind(array); }

protected:
	void allocateFlat(size_t size) override
	{
		(void)size;
		throw std::runtime_error(
		    "Unsupported operation, cannot Allocate on Alias array");
	}
};


// ---------------


template <typename T>
class Array2DBase : public Array<2, T>
{
public:
	Array2DBase() : Array<2, T>()
	{
		size_t dims[2] = {0, 0};
		this->setShape(dims);
	}

	T* operator[](size_t ri) const
	{
		return &this->_data[ri * this->_shape.get()[1]];
	}

	T* operator[](size_t ri)
	{
		return &this->_data[ri * this->_shape.get()[1]];
	}

private:
	Array2DBase(const Array2DBase<T>&) = delete;
};

template <typename T>
class Array2D : public Array2DBase<T>
{
public:
	Array2D() : Array2DBase<T>() {}

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
		size_t dims[2] = {num_rows, num_el};
		this->setShape(dims);
	}

private:
	Array2D(const Array2D<T>&) = delete;

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
class Array2DAlias : public Array2DBase<T>
{
public:
	Array2DAlias() : Array2DBase<T>() {}

	void bind(const Array2DBase<T>& array)
	{
		size_t dims[2];
		array.getDims(dims);
		this->setShape(dims);
		this->_data = array[0];
	}

	void bind(T* data, size_t num_rows, size_t num_el)
	{
		size_t dims[2] = {num_rows, num_el};
		this->setShape(dims);
		this->_data = data;
	}

	Array2DAlias(const Array2DAlias<T>& array) : Array2DBase<T>()
	{
		bind(array);
	}

	Array2DAlias(const Array2DBase<T>* array) { bind(*array); }
	Array2DAlias(const Array2D<T>& array) { bind(array); }

protected:
	void allocateFlat(size_t size) override
	{
		(void)size;
		throw std::runtime_error(
		    "Unsupported operation, cannot Allocate on Alias array");
	}
};


template <typename T>
class Array3DBase : public Array<3, T>
{
public:
	Array3DBase() : Array<3, T>()
	{
		size_t dims[3] = {0, 0, 0};
		this->setShape(dims);
	}

	T* getSlicePtr(size_t ri)
	{
		return &this->_data[ri * this->_shape.get()[1] * this->_shape.get()[2]];
	}

	T* getSlicePtr(size_t ri) const
	{
		return &this->_data[ri * this->_shape.get()[1] * this->_shape.get()[2]];
	}

	Array2DAlias<T> operator[](size_t ri)
	{
		Array2DAlias<T> slice_array;
		T* data_slice = getSlicePtr(ri);
		slice_array.bind(data_slice, this->_shape.get()[1],
		                 this->_shape.get()[2]);
		return slice_array;
	}

	Array2DAlias<T> operator[](size_t ri) const
	{
		Array2DAlias<T> slice_array;
		T* data_slice = getSlicePtr(ri);
		slice_array.bind(data_slice, this->_shape.get()[1],
		                 this->_shape.get()[2]);
		return slice_array;
	}

private:
	Array3DBase(const Array3DBase<T>&) = delete;
};

template <typename T>
class Array3D : public Array3DBase<T>
{
public:
	Array3D() : Array3DBase<T>() {}

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
		size_t dims[3] = {num_slices, num_rows, num_el};
		this->setShape(dims);
	}

private:
	Array3D(const Array3D<T>&) = delete;

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
class Array3DAlias : public Array3DBase<T>
{
public:
	Array3DAlias() : Array3DBase<T>() {}

	void bind(const Array3DBase<T>& array)
	{
		size_t dims[3];
		array.getDims(dims);
		this->setShape(dims);
		this->_data = array[0][0];
	}

	void bind(T* data, size_t num_slices, size_t num_rows, size_t num_el)
	{
		size_t dims[3] = {num_slices, num_rows, num_el};
		this->setShape(dims);
		this->_data = data;
	}

	Array3DAlias(const Array3DBase<T>* array) { bind(*array); }

	Array3DAlias(const Array3DAlias<T>& array) : Array3DBase<T>()
	{
		bind(array);
	}

	Array3DAlias(const Array3D<T>& array) { bind(array); }

protected:
	void allocateFlat(size_t size) override
	{
		(void)size;
		throw std::runtime_error(
		    "Unsupported operation, cannot Allocate on Alias array");
	}
};

}  // namespace yrt
