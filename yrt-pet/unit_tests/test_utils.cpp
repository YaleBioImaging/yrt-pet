#include "test_utils.hpp"

std::unique_ptr<Scanner> TestUtils::makeScanner()
{
	// Fake small scanner
	auto scanner = std::make_unique<Scanner>("FakeScanner", 200, 1, 1, 10, 200,
	                                         24, 9, 2, 4, 6, 4);
	const auto detRegular = std::make_shared<DetRegular>(scanner.get());
	detRegular->generateLUT();
	scanner->setDetectorSetup(detRegular);

	// Sanity check
	if (!scanner->isValid())
	{
		throw std::runtime_error("Unknown error in TestUtils::makeScanner");
	}

	return scanner;
}
