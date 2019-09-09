#include <taskflow/taskflow.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <fmt/core.h>

#include <cctype>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <regex>
#include <vector>
#include <string>


namespace fs = std::filesystem;

auto getFeatures(std::string const& image_name)  -> std::pair<std::vector<cv::KeyPoint>, cv::Mat> {
	auto image = cv::imread(image_name);
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptor;
	auto orb_feature = cv::ORB::create();
	orb_feature->detectAndCompute(image, cv::noArray(), keypoints, descriptor);
	return {keypoints, descriptor};
}

/**
 * \brief compare two string alphanumerically
 * \param first_string
 * \param second_string
 * \return if first_string< second_string
 */
inline bool compare_alphanumeric(const std::string& first_string, const std::string& second_string) {
	if (first_string.empty())
		return true;
	if (second_string.empty())
		return false;
	if (std::isdigit(first_string[0]) && !std::isdigit(second_string[0]))
		return true;
	if (!std::isdigit(first_string[0]) && std::isdigit(second_string[0]))
		return false;
	if (!std::isdigit(first_string[0]) && !std::isdigit(second_string[0])) {
		if (std::toupper(first_string[0]) == std::toupper(second_string[0]))
			return compare_alphanumeric(first_string.substr(1), second_string.substr(1));
		return (std::toupper(first_string[0]) < std::toupper(second_string[0]));
	}

	// Both strings begin with digit --> parse both numbers
	std::istringstream issFirstString(first_string);
	std::istringstream issSecondString(second_string);
	long long ia, ib;
	issFirstString >> ia;
	issSecondString >> ib;
	if (ia != ib)
		return ia < ib;

	// Numbers are the same --> remove numbers and recurse
	std::string anew, bnew;
	std::getline(issFirstString, anew);
	std::getline(issSecondString, bnew);
	return (compare_alphanumeric(anew, bnew));
}

/**
 * \brief
 * \param folder_path
 * \param extensions
 * \param file_filter
 * \param images_vector
 */
void iterate_folder(const std::string& folder_path,
	const std::vector<std::string>& extensions,
	const std::string& file_filter,
	std::vector<std::string>& images_vector) {
	const std::regex extensionFilter(file_filter);
	for (const auto& entry : fs::directory_iterator(folder_path)) {
		const auto& imagePath = entry.path();
		auto extensionName = imagePath.extension().string();
		std::transform(extensionName.begin(), extensionName.end(), extensionName.begin(), ::tolower);
		std::for_each(extensions.begin(), extensions.end(), [&](const std::string& extension) {
			if ((extensionName == extension) 
				&& fs::is_regular_file(imagePath) 
				&& std::regex_match(imagePath.string(), extensionFilter)) {
				images_vector.emplace_back(imagePath.string());
			}
		});
	}
}
/**
 * \brief
 * \param folder_path
 * \param extensions
 * \param file_filter
 * \return
 */
std::vector<std::string> read_image_folder(const std::string& folder_path,
	const std::vector<std::string>& extensions = std::vector<std::string>{ ".jpg", ".png", ".hevc" },
	const std::string& file_filter = ".*") {
	std::vector<std::string> imagesVector;
	iterate_folder(folder_path, extensions, file_filter, imagesVector);
	std::sort(imagesVector.begin(), imagesVector.end(), [](const std::string& first, const std::string& second) {
		const auto firstFileName = fs::path(first).filename().stem().string();
		const auto secondFileName = fs::path(second).filename().stem().string();
		return compare_alphanumeric(firstFileName, secondFileName);
	});
	return imagesVector;
}


void main() {
	tf::Executor executor;
	tf::Taskflow taskflow;
	std::vector<std::string> images = read_image_folder(R"(C:\3DReconTests\ForwardReturn\Images\)");
	fmt::print("Images Number = {}", images.size());
	std::vector<cv::Mat> descriptors(images.size());
	std::vector<std::vector<cv::KeyPoint>> keypoints(images.size());
	
	tf::Task S = taskflow.emplace([]() {}).name("START");
	tf::Task T = taskflow.emplace([]() {}).name("END");
	for (auto i = 0; i < images.size(); i++) {
		tf::Task task1 = taskflow.emplace([i, &images, &keypoints, &descriptors]() {
			auto [keypoint, descriptor] = getFeatures(images[i]);
			keypoints[i] = keypoint;
			descriptors[i] = descriptor;
		}).name("Compute Features");

		tf::Task task2 = taskflow.emplace([i, &keypoints, &descriptors]() {  // create a output task
			cv::FileStorage fs(fmt::format("Keypoints{}.yml",i), cv::FileStorage::WRITE);
			write(fs, fmt::format("keypoints_{}",i), keypoints[i]);
			write(fs, fmt::format("descriptors_{}",i), descriptors[i]);
			fs.release();
		}).name("Write To Disk");
		S.precede(task1);
		task1.precede(task2);
		task2.precede(T);
	}

	taskflow.dump(std::cout);
	executor.run(taskflow).get();
}
