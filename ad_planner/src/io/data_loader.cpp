#include "ad_planner/io/data_loader.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ad_planner {
namespace {

std::string trim(const std::string &input) {
  const auto first = input.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = input.find_last_not_of(" \t\r\n");
  return input.substr(first, last - first + 1);
}

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> fields;
  std::size_t begin = 0;
  while (true) {
    const auto comma = line.find(',', begin);
    if (comma == std::string::npos) {
      fields.push_back(trim(line.substr(begin)));
      return fields;
    }
    fields.push_back(trim(line.substr(begin, comma - begin)));
    begin = comma + 1;
  }
}

std::vector<std::string> split_whitespace(const std::string &line) {
  std::istringstream input(line);
  std::vector<std::string> fields;
  for (std::string field; input >> field;) {
    fields.push_back(std::move(field));
  }
  return fields;
}

[[noreturn]] void parse_error(const std::filesystem::path &path,
                              std::size_t line, const std::string &message) {
  throw std::runtime_error(path.string() + ": line " + std::to_string(line) +
                           ": " + message);
}

double parse_number(const std::string &token, const std::filesystem::path &path,
                    std::size_t line) {
  if (token.empty()) {
    parse_error(path, line, "missing numeric value");
  }
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(token, &consumed);
  } catch (const std::exception &) {
    parse_error(path, line, "invalid numeric value '" + token + "'");
  }
  if (consumed != token.size()) {
    parse_error(path, line, "partial numeric value '" + token + "'");
  }
  if (!std::isfinite(value)) {
    parse_error(path, line, "non-finite numeric value");
  }
  return value;
}

bool equal_points(const Point3 &lhs, const Point3 &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

} // namespace

Route DataLoader::load_path(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open path file: " + path.string());
  }

  std::vector<Point3> source_points;
  bool format_selected = false;
  bool comma_format = false;
  std::string raw_line;
  std::size_t line_number = 0;
  while (std::getline(input, raw_line)) {
    ++line_number;
    const auto comment = raw_line.find('#');
    const std::string line = trim(raw_line.substr(0, comment));
    if (line.empty()) {
      continue;
    }

    const bool line_has_comma = line.find(',') != std::string::npos;
    if (!format_selected) {
      comma_format = line_has_comma;
      format_selected = true;
    } else if (line_has_comma != comma_format) {
      parse_error(path, line_number, "mixed path formats");
    }

    const auto fields = comma_format ? split_csv(line) : split_whitespace(line);
    if (fields.size() < 2 || fields.size() > 3) {
      parse_error(path, line_number, "expected two or three coordinates");
    }
    Point3 point{parse_number(fields[0], path, line_number),
                 parse_number(fields[1], path, line_number),
                 fields.size() == 3 ? parse_number(fields[2], path, line_number)
                                    : 0.0};
    source_points.push_back(point);
  }

  if (source_points.size() < 2) {
    throw std::runtime_error(path.string() +
                             ": path requires at least two usable points");
  }

  std::vector<Point3> points;
  for (const auto &point : source_points) {
    if (points.empty() || !equal_points(points.back(), point)) {
      points.push_back(point);
    }
  }
  if (points.size() < 2) {
    throw std::runtime_error(path.string() + ": fully degenerate path");
  }

  Route route;
  if (equal_points(points.front(), points.back())) {
    route.closed = true;
    points.pop_back();
  }
  if (points.size() < 2) {
    throw std::runtime_error(path.string() + ": fully degenerate path");
  }
  route.points = std::move(points);
  return route;
}

} // namespace ad_planner
