/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "yrt-pet/datastruct/fancyarray/variadic_common.hpp"

#include <fstream>
#include <memory>

namespace fancyarray
{
	template <typename FileIOType>
	void check_file(const FileIOType& file)
	{
		if (!file.is_open())
		{
			throw std::system_error(
			    std::make_error_code(std::errc::no_such_file_or_directory),
			    "Unable to open file");
		}
	}

	template <typename... Types>
	class trivial_struct_of_arrays
	{
	public:
		static constexpr size_t get_num_rows() { return sizeof...(Types); }

		template <size_t row>
		static constexpr size_t get_sizeof()
		{
			return sizeof(get_nth_type_t<row, Types...>);
		}

		static constexpr size_t get_size_per_col()
		{
			return get_total_sizeof_t<Types...>::FinalSum;
		}

		template <size_t row>
		static constexpr size_t get_size_until()
		{
			return get_size_until_t<row, Types...>::FinalSum;
		}

		trivial_struct_of_arrays() : m_num_columns(0) {}

		trivial_struct_of_arrays(const std::string& filename,
		                         const size_t buffer_size_structures = 1024)
		    : trivial_struct_of_arrays()
		{
			read_transpose(filename, buffer_size_structures);
		}

		trivial_struct_of_arrays(const size_t p_num_columns)
		    : m_num_columns(p_num_columns)
		{
			mp_buffer =
			    std::make_unique<char[]>(m_num_columns * get_size_per_col());
		}

		template <size_t row>
		size_t get_size_of_row() const
		{
			return m_num_columns * get_sizeof<row>();
		}

		size_t get_total_size() const
		{
			return m_num_columns * get_size_per_col();
		}

		size_t get_num_columns() const { return m_num_columns; }

		template <size_t row>
		size_t get_total_offset(const size_t col) const
		{
			static_assert(row < get_num_rows(), "Row index out of bounds");
			const size_t offset_row = get_size_until<row>() * m_num_columns;
			const size_t offset_col = get_sizeof<row>() * col;
			return offset_row + offset_col;
		}

		template <size_t row>
		get_nth_type_t<row, Types...> get(const size_t col) const
		{
			return *reinterpret_cast<get_nth_type_t<row, Types...>*>(
			    mp_buffer.get() + get_total_offset<row>(col));
		}

		template <size_t row>
		get_nth_type_t<row, Types...>* get_pointer() const
		{
			static_assert(row < get_num_rows(), "Row index out of bounds");
			return reinterpret_cast<get_nth_type_t<row, Types...>*>(
			    mp_buffer.get() + get_size_until<row>() * m_num_columns);
		}

		template <size_t row>
		void set(const size_t col, get_nth_type_t<row, Types...> value)
		{
			*reinterpret_cast<get_nth_type_t<row, Types...>*>(
			    mp_buffer.get() + get_total_offset<row>(col)) = value;
		}

		void read_transpose(const std::string& fname,
		                    const size_t buffer_size_structures = 1024)
		{
			std::ifstream file(fname, std::ios::binary | std::ios::ate);
			check_file(file);

			constexpr size_t struct_size = get_size_per_col();
			const std::streamsize file_size = file.tellg();
			if (file_size % struct_size != 0)
			{
				throw std::runtime_error("The input file is misformed");
			}
			const size_t total_structures = file_size / struct_size;
			if (total_structures != m_num_columns)
			{
				m_num_columns = total_structures;
				mp_buffer =
				    std::make_unique<char[]>(m_num_columns * struct_size);
			}

			// Intermediary buffer
			const size_t buffer_size = buffer_size_structures * struct_size;
			const auto buffer = std::make_unique<char[]>(buffer_size);

			file.seekg(0, std::ios::beg);
			size_t col = 0;
			while (file.read(buffer.get(), buffer_size) || file.gcount() > 0)
			{
				const size_t num_read_structures = file.gcount() / struct_size;
				for (size_t i = 0;
				     i < num_read_structures && col < m_num_columns; ++i, ++col)
				{
					size_t offset = 0;
					transpose_helper<0>(buffer.get() + i * struct_size, col,
					                    offset);
				}
			}

			file.close();
		}

		void save_transpose(const std::string& fname,
		                    const size_t buffer_size_structures = 1024) const
		{
			std::ofstream file(fname, std::ios::binary);
			check_file(file);

			constexpr size_t struct_size = get_size_per_col();
			const size_t buffer_size = buffer_size_structures * struct_size;
			const auto buffer = std::make_unique<char[]>(buffer_size);

			size_t col = 0;
			while (col < m_num_columns)
			{
				const size_t num_structures_in_buffer =
				    std::min(buffer_size, m_num_columns - col);
				for (size_t i = 0; i < num_structures_in_buffer; ++i, ++col)
				{
					size_t offset = 0;
					assemble_structure<0>(buffer.get() + i * struct_size, col,
					                      offset);
				}

				file.write(buffer.get(),
				           num_structures_in_buffer * struct_size);
			}

			file.close();
		}

	private:
		template <size_t row>
		void transpose_helper(const char* struct_buffer, const size_t col,
		                      size_t& offset)
		{
			using T = get_nth_type_t<row, Types...>;
			const T value = *reinterpret_cast<const T*>(struct_buffer + offset);

			set<row>(col, value);
			offset += sizeof(T);

			if constexpr (row + 1 < get_num_rows())
			{
				transpose_helper<row + 1>(struct_buffer, col, offset);
			}
		}

		template <size_t row>
		void assemble_structure(char* struct_buffer, const size_t col,
		                        size_t& offset) const
		{
			using T = get_nth_type_t<row, Types...>;
			T value = get<row>(col);

			*reinterpret_cast<T*>(struct_buffer + offset) = value;
			offset += sizeof(T);

			if constexpr (row + 1 < get_num_rows())
			{
				assemble_structure<row + 1>(struct_buffer, col, offset);
			}
		}

		std::unique_ptr<char[]> mp_buffer;
		size_t m_num_columns;
	};

}  // namespace fancyarray
