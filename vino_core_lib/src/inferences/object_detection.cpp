/*
 * Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @brief a header file with declaration of ObjectDetection class and
 * ObjectDetectionResult class
 * @file object_detection.cpp
 */
#include <memory>
#include <string>
#include <vector>
#include "vino_core_lib/inferences/object_detection.h"
#include "vino_core_lib/outputs/base_output.h"
#include "vino_core_lib/slog.h"
// ObjectDetectionResult
vino_core_lib::ObjectDetectionResult::ObjectDetectionResult(
    const cv::Rect& location)
    : Result(location){}
// ObjectDetection
vino_core_lib::ObjectDetection::ObjectDetection(bool enable_roi_constraint, 
                                                  double show_output_thresh)
    : vino_core_lib::BaseInference(),
      show_output_thresh_(show_output_thresh){}
vino_core_lib::ObjectDetection::~ObjectDetection() = default;
void vino_core_lib::ObjectDetection::loadNetwork(
    const std::shared_ptr<Models::ObjectDetectionModel> network) {
  valid_model_ = network;
  max_proposal_count_ = network->getMaxProposalCount();
  object_size_ = network->getObjectSize();
  setMaxBatchSize(network->getMaxBatchSize());
}
bool vino_core_lib::ObjectDetection::enqueue(const cv::Mat& frame,
                                              const cv::Rect& input_frame_loc) {
  if (width_ == 0 && height_ == 0) {
    width_ = frame.cols;
    height_ = frame.rows;
  }
  if (!vino_core_lib::BaseInference::enqueue<u_int8_t>(
          frame, input_frame_loc, 1, 0, valid_model_->getInputName())) {
    return false;
  }
  Result r(input_frame_loc);
  results_.clear();
  results_.emplace_back(r);
  return true;
}
bool vino_core_lib::ObjectDetection::submitRequest() {
  return vino_core_lib::BaseInference::submitRequest();
}
//进入对应的index块
int vino_core_lib::ObjectDetection::EntryIndex(int side, int lcoords, int lclasses, int location, int entry){
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

bool vino_core_lib::ObjectDetection::fetchResults() {
  bool can_fetch = vino_core_lib::BaseInference::fetchResults();
  if (!can_fetch) return false;
  bool found_result = false;
  results_.clear();
  InferenceEngine::InferRequest::Ptr request = getEngine()->getRequest();
  std::string output = valid_model_->getOutputName();
  const float* detections = request->GetBlob(output)->buffer().as<float*>();

  //get required parameter
  int num = 3;
  int coords = 4;
  int classes = 80;
  std::vector<float> anchors = { 30.0, 61.0, 62.0, 45.0, 59.0, 119.0};
  int side =26;
  int side_square = side * side;
  int resized_im_w = 416;
  int resized_im_h = 416;
  double threshold = 0.1;
  double x, y, width, height;
  float w_scale = static_cast<float>(width_) / static_cast<float>(resized_im_w);               
  float h_scale = static_cast<float>(height_) / static_cast<float>(resized_im_h);
  int label_num = 0;
  cv::Rect r;
  std::vector<std::string>& labels = valid_model_->getLabels();
  /*
  if (anchors.size() == 18) {        // YoloV3
    switch (side) {
        case yolo_scale_13:
            anchor_offset = 2 * 6;
            break;
        case yolo_scale_26:
            anchor_offset = 2 * 3;
            break;
        case yolo_scale_52:
            anchor_offset = 2 * 0;
            break;
        default:
            throw std::runtime_error("Invalid output size");
    }
  */
  InferenceEngine::Blob::Ptr blob = request->GetBlob(output);
  const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
  const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
  slog::info << "out_blob_h: " << out_blob_h << " out_blob_w" << out_blob_w << slog::endl;

  //refer for these code

  for (int i = 0; i < side_square; ++i) {
    int row = i / side;
    int col = i % side;
    for (int n = 0; n < num; ++n) {
        int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
        int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
        float scale = detections[obj_index];
        if (scale < threshold)
            continue;
        x = (col + detections[box_index + 0 * side_square]) / side * resized_im_w;
        y = (row + detections[box_index + 1 * side_square]) / side * resized_im_h;
        height = std::exp(detections[box_index + 3 * side_square]) * anchors[2 * n + 1];
        width = std::exp(detections[box_index + 2 * side_square]) * anchors[2 * n];
        for (int j = 0; j < classes; ++j) {
            int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
            float prob = scale * detections[class_index];
            if (prob < threshold)
                continue;
            r.x = static_cast<int>((x - width / 2) * w_scale);
            r.y = static_cast<int>((y - height / 2) * h_scale);
            r.width = static_cast<int>(width * w_scale);
            r.height = static_cast<int>(height * h_scale);
            Result result(r);
            label_num = j;
            result.label_ = label_num < 80
                        ? labels[label_num]
                        : std::string("label #") + std::to_string(label_num);
            found_result = true;
            results_.emplace_back(result);
        }
    }
  }
  if (!found_result) results_.clear();
  return true;
}
const int vino_core_lib::ObjectDetection::getResultsLength() const {
  return static_cast<int>(results_.size());
}
const vino_core_lib::Result*
vino_core_lib::ObjectDetection::getLocationResult(int idx) const {
  return &(results_[idx]);
}
const std::string vino_core_lib::ObjectDetection::getName() const {
  return valid_model_->getModelName();
}
const void vino_core_lib::ObjectDetection::observeOutput(
    const std::shared_ptr<Outputs::BaseOutput>& output) {
  if (output != nullptr) {
    output->accept(results_);
  }
}
