/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <exception>

namespace yrt
{

void assertion_failed(char const* expr, char const* function, char const* file,
                      long line, bool critical);

void assertion_failed_msg(char const* expr, char const* msg,
                          char const* function, char const* file, long line,
                          bool critical);


#define CHECK_LIKELY(x) __builtin_expect(x, 1)

/* clang-format off */
#define ASSERT(expr) (CHECK_LIKELY(!!(expr))? ((void)0): yrt::assertion_failed(#expr, __PRETTY_FUNCTION__, __FILE__, __LINE__, true))
#define ASSERT_MSG(expr, msg) (CHECK_LIKELY(!!(expr))? ((void)0): yrt::assertion_failed_msg(#expr, msg, __PRETTY_FUNCTION__, __FILE__, __LINE__, true))
#define ASSERT_WARNING(expr) (CHECK_LIKELY(!!(expr))? ((void)0): yrt::assertion_failed(#expr, __PRETTY_FUNCTION__, __FILE__, __LINE__, false))
#define ASSERT_MSG_WARNING(expr, msg) (CHECK_LIKELY(!!(expr))? ((void)0): yrt::assertion_failed_msg(#expr, msg, __PRETTY_FUNCTION__, __FILE__, __LINE__, false))
/* clang-format on */

namespace util
{
void printExceptionMessage(const std::exception& e);
}

}  // namespace yrt
