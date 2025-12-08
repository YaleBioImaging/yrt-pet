/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */
namespace yrt
{
// Note: FLAG_SIDDON_INCR skips the conversion from physical to logical
// coordinates by moving from pixel to pixel as the ray parameter is updated.
// This may cause issues near the last intersection, which must therefore be
// handled with extra care.  Speedups around 20% were measured with
// FLAG_SIDDON_INCR=true.  Both versions are compared in tests, the "faster"
// version (FLAG_SIDDON_INCR=true) is used by default.
constexpr bool FLAG_SIDDON_INCR = true;
constexpr float EPS_SIDDON = 1.0e-6f;
enum SIDDON_DIR
{
	DIR_X = 0b001,
	DIR_Y = 0b010,
	DIR_Z = 0b100
};
}  // namespace yrt
