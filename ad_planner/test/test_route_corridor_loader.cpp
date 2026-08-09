#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "ad_planner/io/route_corridor_loader.hpp"

namespace
{

using ad_planner::load_route_corridor;

class TemporaryFile
{
public:
  explicit TemporaryFile(const std::string & contents)
  {
    static std::size_t sequence = 0;
    path_ = std::filesystem::temp_directory_path() /
      ("ad_planner_route_corridor_" + std::to_string(++sequence) + ".json");
    std::ofstream output(path_);
    output << contents;
  }

  ~TemporaryFile()
  {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  const std::filesystem::path & path() const {return path_;}

private:
  std::filesystem::path path_;
};

const std::string kDigest(64, 'a');

std::string point_document(const std::string & route_s = "0.0", const std::string & x = "0.0")
{
  return R"({"curvature_inv_m":0.0,"left_width_m":1.75,"right_width_m":1.75,
    "route_s_m":)" + route_s + R"(,"speed_limit_mps":13.0,"x_m":)" + x + R"(,
    "y_m":0.0,"yaw_rad":0.0,"z_m":0.0})";
}

std::string lane_document(
  const std::string & id, const std::string & adjacency = "{}",
  const std::string & points = "")
{
  const std::string usable_points = points.empty() ?
    "[" + point_document("0.0") + "," + point_document("2.0", "2.0") + "]" : points;
  return R"({"adjacent_lane_sequence_ids":)" + adjacency + R"(,"lane_sequence_id":")" + id +
    R"(","points":)" + usable_points + R"(,"source_link_attributes":[{"id":"source-)" + id +
    R"(","max_speed":50.0,"width_end":3.5,"width_start":3.5}],"source_link_ids":["source-)" +
    id + R"("]})";
}

std::string document(
  const std::string & primary_adjacency =
    R"({"left":["left:1","left:2"],"right":["right:1","right:2","right:3","right:4","right:5"]})",
  const std::string & primary_points = "")
{
  std::string lanes = lane_document("route:0", primary_adjacency, primary_points);
  lanes += "," + lane_document("left:1", R"({"right":["route:0"]})");
  lanes += "," + lane_document("left:2", R"({"right":["route:0"]})");
  for (int index = 1; index <= 5; ++index) {
    lanes += "," + lane_document(
      "right:" + std::to_string(index), R"({"left":["route:0"]})");
  }
  return R"({"frame_id":"map","lanes":[)" + lanes + R"(],"primary_lane_sequence_id":"route:0",
    "schema_version":1,"source_sha256":{"global_path":")" + kDigest + R"("}})";
}

void expect_rejected(const std::string & contents, const std::string & diagnostic)
{
  TemporaryFile file(contents);
  try {
    static_cast<void>(load_route_corridor(file.path()));
    FAIL() << "expected route corridor to be rejected";
  } catch (const std::runtime_error & error) {
    EXPECT_NE(std::string(error.what()).find(diagnostic), std::string::npos) << error.what();
  }
}

std::string replace_first(
  std::string contents, const std::string & original, const std::string & replacement)
{
  const auto position = contents.find(original);
  if (position == std::string::npos) {
    throw std::logic_error("test document does not contain '" + original + "'");
  }
  contents.replace(position, original.size(), replacement);
  return contents;
}

TEST(RouteCorridorLoader, LoadsSeparateAdjacentLanesAndUnwrappedProgress)
{
  TemporaryFile file(document());

  const auto loaded = load_route_corridor(file.path(), {{"global_path", kDigest}});

  EXPECT_EQ(loaded.metadata.schema_version, 1);
  EXPECT_EQ(loaded.metadata.source_sha256.at("global_path"), kDigest);
  EXPECT_EQ(loaded.corridor.frame_id, "map");
  EXPECT_EQ(loaded.corridor.primary_lane_index, 0U);
  ASSERT_EQ(loaded.corridor.lanes[0].points.size(), 2U);
  EXPECT_DOUBLE_EQ(loaded.corridor.lanes[0].points[1].route_s_m, 2.0);
  EXPECT_EQ(loaded.corridor.lanes[0].source_link_ids, (std::vector<std::string>{"source-route:0"}));
}

TEST(RouteCorridorLoader, RejectsUnknownSchemaVersion)
{
  auto contents = document();
  contents.replace(contents.find("\"schema_version\":1"), 18, "\"schema_version\":2");
  expect_rejected(contents, "schema_version");
}

TEST(RouteCorridorLoader, RejectsDigestMismatch)
{
  TemporaryFile file(document());

  EXPECT_THROW(load_route_corridor(file.path(), {{"global_path", std::string(64, 'b')}}), std::runtime_error);
}

TEST(RouteCorridorLoader, RejectsNonMonotonicRouteProgress)
{
  expect_rejected(document({}, "[" + point_document("1.0") + "," + point_document("0.5") + "]"), "route_s_m");
}

TEST(RouteCorridorLoader, RejectsDanglingAdjacentLaneIndex)
{
  expect_rejected(document(R"({"left":["missing"]})"), "unknown adjacent");
}

TEST(RouteCorridorLoader, PreservesAllDisjointAdjacentLaneSequences)
{
  TemporaryFile file(document());

  const auto loaded = load_route_corridor(file.path());
  const auto & primary = loaded.corridor.lanes.at(loaded.corridor.primary_lane_index);

  EXPECT_EQ(primary.left_lane_indices, (std::vector<std::size_t>{1U, 2U}));
  EXPECT_EQ(primary.right_lane_indices, (std::vector<std::size_t>{3U, 4U, 5U, 6U, 7U}));
}

TEST(RouteCorridorLoader, RejectsNonfiniteGeometry)
{
  expect_rejected(document({}, "[" + point_document("0.0", ".nan") + "," + point_document("2.0", "2.0") + "]"), "finite");
}

TEST(RouteCorridorLoader, RejectsNonfiniteUnusedGeometryCoordinates)
{
  auto contents = document();
  constexpr const char kFiniteZ[] = "\"z_m\":0.0";
  contents.replace(contents.find(kFiniteZ), sizeof(kFiniteZ) - 1U, "\"z_m\":.nan");
  expect_rejected(contents, "z_m");
}

TEST(RouteCorridorLoader, RejectsNonpositiveLeftLaneWidth)
{
  expect_rejected(
    replace_first(document(), R"("left_width_m":1.75)", R"("left_width_m":0.0)"),
    "left_width_m must be greater than zero");
  expect_rejected(
    replace_first(document(), R"("left_width_m":1.75)", R"("left_width_m":-1.0)"),
    "left_width_m must be greater than zero");
}

TEST(RouteCorridorLoader, RejectsNonpositiveRightLaneWidth)
{
  expect_rejected(
    replace_first(document(), R"("right_width_m":1.75)", R"("right_width_m":0.0)"),
    "right_width_m must be greater than zero");
  expect_rejected(
    replace_first(document(), R"("right_width_m":1.75)", R"("right_width_m":-1.0)"),
    "right_width_m must be greater than zero");
}

TEST(RouteCorridorLoader, RejectsNegativeSpeedLimit)
{
  expect_rejected(
    replace_first(document(), R"("speed_limit_mps":13.0)", R"("speed_limit_mps":-0.1)"),
    "speed_limit_mps must not be negative");
}

TEST(RouteCorridorLoader, RejectsDuplicateAndSelfAdjacentLaneReferences)
{
  expect_rejected(document(R"({"left":["left:1","left:1"]})"), "duplicate adjacent");
  expect_rejected(document(R"({"left":["route:0"]})"), "self");
}

TEST(RouteCorridorLoader, RejectsUnexpectedAdjacencySideAndMalformedDigest)
{
  expect_rejected(document(R"({"up":["left:1"]})"), "adjacent_lane_sequence_ids");
  auto contents = document();
  contents.replace(contents.find(kDigest), kDigest.size(), "not-a-sha256");
  expect_rejected(contents, "SHA-256");
}

}  // namespace
