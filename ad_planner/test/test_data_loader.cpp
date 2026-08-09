#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/io/data_loader.hpp"

namespace {

using ad_planner::DataLoader;

class TemporaryFile {
public:
  explicit TemporaryFile(const std::string &contents) {
    static std::size_t sequence = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("ad_planner_test_" + std::to_string(++sequence) + ".txt");
    std::ofstream output(path_);
    output << contents;
    output.close();
  }

  ~TemporaryFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

std::filesystem::path fixture(const std::string &name) {
  return std::filesystem::path("fixtures") / name;
}

void expect_path_error(const std::string &contents,
                       const std::string &diagnostic) {
  TemporaryFile file(contents);
  try {
    static_cast<void>(DataLoader::load_path(file.path()));
    FAIL() << "expected path parsing to fail";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(diagnostic), std::string::npos)
        << error.what();
  }
}

TEST(DataLoaderPath,
     LoadsOneCsvFormatStripsCommentsAndRemovesClosingDuplicate) {
  const auto route = DataLoader::load_path(fixture("path_csv.txt"));

  ASSERT_EQ(route.points.size(), 3U);
  EXPECT_TRUE(route.closed);
  EXPECT_DOUBLE_EQ(route.points[1].x, 2.0);
  EXPECT_DOUBLE_EQ(route.points[1].z, 0.5);
  EXPECT_DOUBLE_EQ(route.points.back().y, 2.0);
}

TEST(DataLoaderPath, LoadsOneWhitespaceFormatAndDefaultsMissingZToZero) {
  const auto route = DataLoader::load_path(fixture("path_space.txt"));

  ASSERT_EQ(route.points.size(), 3U);
  EXPECT_FALSE(route.closed);
  EXPECT_DOUBLE_EQ(route.points[2].x, 2.0);
  EXPECT_DOUBLE_EQ(route.points[2].z, 0.0);
}

TEST(DataLoaderPath, RejectsMixedFileWideFormats) {
  expect_path_error("0,0\n1 1\n", "line 2");
}

TEST(DataLoaderPath, RejectsMissingAndExtraColumnsInEitherFormat) {
  expect_path_error("0,0\n1\n", "line 2");
  expect_path_error("0,0\n1,2,3,4\n", "line 2");
  expect_path_error("0 0\n1\n", "line 2");
  expect_path_error("0 0\n1 2 3 4\n", "line 2");
}

TEST(DataLoaderPath, RejectsEmptyColumnsAndPartialNumericTokens) {
  expect_path_error("0,0\n1,,2\n", "line 2");
  expect_path_error("0 0\n1.0junk 2.0\n", "line 2");
}

TEST(DataLoaderPath, RejectsEveryNonFiniteRepresentation) {
  expect_path_error("0,0\nnan,1\n", "line 2");
  expect_path_error("0 0\ninf 1\n", "line 2");
  expect_path_error("0 0\n-inf 1\n", "line 2");
}

TEST(DataLoaderPath, RejectsFewerThanTwoUsablePointsAndFullyDegeneratePaths) {
  expect_path_error("\n# no usable points\n", "at least two");
  expect_path_error("0,0\n", "at least two");
  expect_path_error("0,0\n0,0\n", "degenerate");
}

TEST(DataLoaderPath, IgnoresConsecutiveDuplicates) {
  TemporaryFile file("0,0\n1,0\n1,0\n2,0\n");

  const auto route = DataLoader::load_path(file.path());

  ASSERT_EQ(route.points.size(), 3U);
  EXPECT_DOUBLE_EQ(route.points[1].x, 1.0);
  EXPECT_DOUBLE_EQ(route.points[2].x, 2.0);
}

TEST(DataLoaderPath, ClosingRuleStoresEachCoordinateOnce) {
  TemporaryFile file("0,0\n1,0\n0,0\n0,0\n");

  const auto route = DataLoader::load_path(file.path());

  ASSERT_EQ(route.points.size(), 2U);
  EXPECT_TRUE(route.closed);
  EXPECT_NE(route.points.front().x, route.points.back().x);
}

TEST(DataLoaderPath, ReportsMissingFileAsExplicitError) {
  EXPECT_THROW(DataLoader::load_path("fixtures/does-not-exist.txt"),
               std::runtime_error);
}

} // namespace
