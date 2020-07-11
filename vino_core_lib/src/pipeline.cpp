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
 * @brief a header file with declaration of Pipeline class
 * @file pipeline.cpp
 */

#include <memory>
#include <string>
#include <utility>

#include <vino_param_lib/param_manager.h>
#include "vino_core_lib/pipeline.h"

Pipeline::Pipeline(const std::string& name)
{
  params_ = std::make_shared<PipelineParams>(name);
  counter_ = 0;
}

bool Pipeline::add(const std::string& name,
                   std::shared_ptr<Input::BaseInputDevice> input_device)
{
  input_device_name_ = name;
  input_device_ = input_device;
  next_.insert({"", name});
  return true;
}

bool Pipeline::add(const std::string& parent, const std::string& name,
                   std::shared_ptr<Outputs::BaseOutput> output)
{
  if (parent.empty())
  {
    slog::err << "output device have no parent!" << slog::endl;
    return false;
  }

  if (name_to_detection_map_.find(parent) == name_to_detection_map_.end())
  {
    slog::err << "parent detection does not exists!" << slog::endl;
    return false;
  }

  if (output_names_.find(name) != output_names_.end())
  {
    return add(parent, name);
  }

  output_names_.insert(name);
  name_to_output_map_[name] = output;
  next_.insert({parent, name});

  /**< Add pipeline instance to Output instance >**/
  output->setPipeline(this);
  return true;
}

bool Pipeline::add(const std::string& parent, const std::string& name)
{
  if (isLegalConnect(parent, name)) {
    addConnect(parent, name);
    return true;
  }

  return false;
}

bool Pipeline::add(const std::string& name,
                   std::shared_ptr<Outputs::BaseOutput> output) {
    if (name.empty()) {
    slog::err << "Item name can't be empty!" << slog::endl;
    return false;
  }
  
  std::map<std::string, std::shared_ptr<Outputs::BaseOutput>>::iterator it = name_to_output_map_.find(name);
  if (it != name_to_output_map_.end()) {
    slog::warn << "inferance instance for [" << name << 
                  "] already exists, update it with new instance." << slog::endl;
  }
  name_to_output_map_[name] = output;
  output_names_.insert(name);
  /**< Add pipeline instance to Output instance >**/
  output->setPipeline(this);
  
  return true;
}

bool Pipeline::add(const std::string& parent, const std::string& name,
                   std::shared_ptr<vino_core_lib::BaseInference> inference)
{
  if (name_to_detection_map_.find(parent) == name_to_detection_map_.end() &&
      input_device_name_ != parent)
  {
    slog::err << "parent device/detection does not exists!" << slog::endl;
    return false;
  }
  next_.insert({parent, name});
  name_to_detection_map_[name] = inference;
  ++total_inference_;
  return true;
}

bool Pipeline::add(const std::string& name,
                   std::shared_ptr<vino_core_lib::BaseInference> inference) {
  if (name.empty()) {
    slog::err << "Item name can't be empty!" << slog::endl;
    return false;
  }
  
  std::map<std::string, std::shared_ptr<vino_core_lib::BaseInference>>::iterator it = name_to_detection_map_.find(name);
  if (it != name_to_detection_map_.end()) {
    slog::warn << "inferance instance for [" << name << 
                  "] already exists, update it with new instance." << slog::endl;
  } else {
    ++total_inference_;
  }
  name_to_detection_map_[name] = inference;

  return true;
}

int Pipeline::getCatagoryOrder(const std::string name)
{
  int order = kCatagoryOrder_Unknown;
  if (name == input_device_name_) {
    order = kCatagoryOrder_Input;
  } else if (name_to_detection_map_.find(name) != name_to_detection_map_.end()) {
    order = kCatagoryOrder_Inference;
  } else if (name_to_output_map_.find(name) != name_to_output_map_.end()) {
    order = kCatagoryOrder_Output;
  }

  return order;
}


bool Pipeline::isLegalConnect(const std::string parent, const std::string child)
{
  int parent_order = getCatagoryOrder(parent);
  int child_order = getCatagoryOrder(child);
  slog::info << "Checking connection into pipeline:[" << parent << "(" << parent_order << ")" <<
    "<-->" << child << "(" << child_order << ")" <<
    "]" << slog::endl;

  return (parent_order != kCatagoryOrder_Unknown) && (child_order != kCatagoryOrder_Unknown) &&
         (parent_order <= child_order);
}

void Pipeline::addConnect(const std::string & parent, const std::string & name)
{
  std::pair<std::multimap<std::string, std::string>::iterator,
    std::multimap<std::string, std::string>::iterator>
  ret;
  ret = next_.equal_range(parent);

  for (std::multimap<std::string, std::string>::iterator it = ret.first; it != ret.second; ++it) {
    if (it->second == name) {
      slog::warn << "The connect [" << parent << "<-->" << name << "] already exists." <<
        slog::endl;
      return;
    }
  }
  slog::info << "Adding connection into pipeline:[" << parent << "<-->" << name << "]" <<
    slog::endl;
  next_.insert({parent, name});
}


void Pipeline::runOnce()
{
  initInferenceCounter();

  if (!input_device_->read(&frame_))
  {
    // throw std::logic_error("Failed to get frame from cv::VideoCapture");
    slog::warn << "Failed to get frame from input_device." << slog::endl;
    return;
  }

  countFPS();
  width_ = frame_.cols;
  height_ = frame_.rows;
  for (auto& pair : name_to_output_map_)
  {
    pair.second->feedFrame(frame_);
  }
  // auto t0 = std::chrono::high_resolution_clock::now();
  for (auto pos = next_.equal_range(input_device_name_);
       pos.first != pos.second; ++pos.first)
  {
    std::string detection_name = pos.first->second;
    auto detection_ptr = name_to_detection_map_[detection_name];
    detection_ptr->enqueue(frame_,
                           cv::Rect(width_ / 2, height_ / 2, width_, height_));
    increaseInferenceCounter();
    detection_ptr->submitRequest();
  }
  std::unique_lock<std::mutex> lock(counter_mutex_);
  cv_.wait(lock, [this]() { return this->counter_ == 0; });
  // auto t1 = std::chrono::high_resolution_clock::now();
  // typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

  for (auto& pair : name_to_output_map_)
  {
    pair.second->handleOutput();
  }
}

void Pipeline::printPipeline()
{
  for (auto& current_node : next_)
  {
    printf("%s --> %s\n", current_node.first.c_str(),
           current_node.second.c_str());
  }
}

void Pipeline::setCallback()
{
  for (auto& pair : name_to_detection_map_)
  {
    std::string detection_name = pair.first;
    std::function<void(void)> callb;
    callb = [detection_name, this]()
    {
      this->callback(detection_name);
      return;
    };
    pair.second->getEngine()->getRequest()->SetCompletionCallback(callb);
  }
}
void Pipeline::callback(const std::string& detection_name)
{
  auto detection_ptr = name_to_detection_map_[detection_name];
  detection_ptr->fetchResults();
  // set output
  for (auto pos = next_.equal_range(detection_name); pos.first != pos.second;
       ++pos.first)
  {
    std::string next_name = pos.first->second;
    // if next is output, then print
    if (output_names_.find(next_name) != output_names_.end())
    {
      // name_to_output_map_[next_name]->accept(*detection_ptr->getResult());
      detection_ptr->observeOutput(name_to_output_map_[next_name]);
    }
    else
    {
      auto detection_ptr_iter = name_to_detection_map_.find(next_name);
      if (detection_ptr_iter != name_to_detection_map_.end())
      {
        auto next_detection_ptr = detection_ptr_iter->second;
        for (int i = 0; i < detection_ptr->getResultsLength(); ++i)
        {
          const vino_core_lib::Result* prev_result =
              detection_ptr->getLocationResult(i);
          auto clippedRect =
              prev_result->getLocation() & cv::Rect(0, 0, width_, height_);
          cv::Mat next_input = frame_(clippedRect);
          next_detection_ptr->enqueue(next_input, prev_result->getLocation());
        }
        if (detection_ptr->getResultsLength() > 0)
        {
          increaseInferenceCounter();
          next_detection_ptr->submitRequest();
        }
      }
    }
  }

  decreaseInferenceCounter();
  cv_.notify_all();
}

void Pipeline::initInferenceCounter()
{
  std::lock_guard<std::mutex> lk(counter_mutex_);
  counter_ = 0;
  cv_.notify_all();
}
void Pipeline::increaseInferenceCounter()
{
  std::lock_guard<std::mutex> lk(counter_mutex_);
  ++counter_;
}
void Pipeline::decreaseInferenceCounter()
{
  std::lock_guard<std::mutex> lk(counter_mutex_);
  --counter_;
}

void Pipeline::countFPS()
{
  static int fps = 0;

  static auto t_start = std::chrono::high_resolution_clock::now();
  static int frame_cnt = 0;

  frame_cnt++;

  auto t_end = std::chrono::high_resolution_clock::now();
  typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
  ms secondDetection = std::chrono::duration_cast<ms>(t_end - t_start);

  if (secondDetection.count() > 1000) {
    fps = frame_cnt;
    setFPS(fps);
    frame_cnt = 0;
    t_start = t_end;
  }

} 