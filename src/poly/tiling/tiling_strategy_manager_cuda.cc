/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tiling_strategy_manager.h"

#include <numeric>

#include "tiling_analyzer.h"

namespace akg {
namespace ir {
namespace poly {

void ReduceStrategy::AddGpuConstraint() {
  // TODO: compare XLA's reduction tiling/mapping strategy with current strategy
  auto reduce_axes = analyzer_->GetAxesOfAttr("REDUCE_AXIS");
  size_t depth = 0;
  bool has_transpose = false;
  analyzer_->ForEachAxisTopDown([this, &depth, &reduce_axes, &has_transpose](TileAxis *axis) {
    if (!has_transpose) {
      for (const auto &attr : axis->attrs) {
        if (attr.attr_key.find("TRANSPOSE") != std::string::npos) {
          has_transpose = true;
          break;
        }
      }
    }

    if (axis == analyzer_->RootAxis()) {
      return;
    }
    ++depth;

    if (axis->mc_sup) {
      return;
    }
    bool already_added = false;
    for (const auto &ra : reduce_axes) {
      if (ra == axis) {
        already_added = true;
        break;
      }
    }
    if (!already_added) {
      reduce_axes.emplace_back(axis);
    }
  });
  bool all_reduce = reduce_axes.size() == depth;
  if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
    bool disable_atomic = !analyzer_->GetAxesOfAttr("CAST").empty();
    if (disable_atomic) {
      for (auto axis : reduce_axes) {
        axis->block_constraints.map_extent_ = MIN_TILE;
      }
    }
    if (!all_reduce) {
      DealWith4DFusedReduce(reduce_axes);
    }
  } else if (all_reduce || has_transpose) {
    auto extent = all_reduce ? MIN_TILE : warp_sizes_;
    bool is_tuning = analyzer_->scop_info_.user_config_.GetIsTuning();
    for (auto axis : reduce_axes) {
      axis->block_constraints.map_extent_ = MIN_TILE;
      axis->thread_constraints.map_extent_ = MIN_TILE;
      if (!is_tuning) {
        axis->TileRestrainToSingleValue(CastIntToExpr(extent), TileLevel::LEVEL1);
      }
    }
  }
}

void ReduceStrategy::DealWith4DFusedReduce(const std::vector<akg::ir::poly::TileAxis *> &reduce_axes) {
  auto mod_axes = analyzer_->GetAxesOfAttr("MOD");

  for (auto axis : mod_axes) {
    if (std::count(reduce_axes.begin(), reduce_axes.end(), axis)) {
      continue;
    }
    int last_mod_value = -1;
    size_t num_mod_axis = 0;
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != "MOD") {
        continue;
      }
      CHECK_NE(attr.attr_value, "");
      last_mod_value = static_cast<int>(std::strtol(attr.attr_value.c_str(), nullptr, 10));
      ++num_mod_axis;
    }
    if (num_mod_axis < 2) {
      continue;
    }
    axis->TileRestrainToSingleValue(CastIntToExpr(last_mod_value), TileLevel::LEVEL1);
    if (last_mod_value > max_num_threads_) {
      LOG(WARNING) << "Cannot bind axis to " << last_mod_value << " threads, maximal thread number is "
                   << max_num_threads_
                   << ". If fusing more than two axes together, footprint box calculated by isl may not be correct.";
      continue;
    }
    axis->thread_constraints.map_min_ = last_mod_value;
    axis->thread_constraints.map_extent_ = last_mod_value;
  }
}

void GpuStrategy::AddGpuConstraint() {
  InitMappingLimit();
  BuildAxesQueue();
  if (analyzer_->scop_info_.user_config_.GetIsTuning()) {
    return;
  }
  InnerThreadOuterBlock();
  SetMappingConfig();
}

void GpuStrategy::InitMappingLimit() {
  DetermineTemplate();
  std::stringstream ss;
  ss << "Use template " << template_;
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
  auto thread_config = analyzer_->scop_info_.user_config_.GetThreadConfig();
  if (thread_config == nullptr || thread_config->bound == 0) {
    if (template_ <= Template::REDUCTION || template_ == Template::BITWISE_REDUCTION) {
      thread_limit_ = {max_num_threads_, max_num_threads_};
    } else if (template_ == Template::ALL_REDUCE) {
      if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
        thread_limit_ = {max_num_threads_, max_num_threads_};
      } else {
        thread_limit_ = {1};
      }
    } else if (template_ == Template::TRANSPOSE) {
      // This is a naive tiling strategy used in gpu when thread and block configs are already set.
      // This strategy will tile up to three inner-most axes to 32 (for thread binding).
      thread_limit_ = {32, 8};
    }
    AdjustThreadMappingLimit();
  } else {
    for (size_t i = 0; i < thread_config->bound; ++i) {
      thread_limit_.emplace_back(thread_config->GetAt(i).second);
    }
  }

  auto block_config = analyzer_->scop_info_.user_config_.GetBlockConfig();
  if (block_config == nullptr || block_config->bound == 0) {
    if (template_ <= Template::REDUCTION) {
      block_limit_ = {max_num_blocks_, max_num_blocks_, max_num_blocks_};
    } else if (template_ == Template::ALL_REDUCE) {
      if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
        block_limit_ = {max_num_blocks_, max_num_blocks_, max_num_blocks_};
      } else {
        block_limit_ = {1};
      }
    } else if (template_ == Template::TRANSPOSE) {
      block_limit_ = {max_num_blocks_, max_num_blocks_};
    } else if (template_ == Template::BITWISE_REDUCTION) {
      if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
        block_limit_ = {1};
      } else {
        block_limit_ = {max_num_blocks_, max_num_blocks_, max_num_blocks_};
      }
    }
  } else {
    for (int i = block_config->bound - 1; i >= 0; --i) {
      block_limit_.emplace_back(block_config->GetAt(i).second);
    }
  }
}

void GpuStrategy::BuildAxesQueue() {
  analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
    if (axis == this->analyzer_->RootAxis()) {
      return;
    }
    const auto r = axis->range_extent.as<IntImm>();
    if (r && r->value > 0) {
      this->pending_axes_.push_front(std::make_pair(axis, r->value));
    }

    // init map extent to shape if they are not modified by other constraints
    axis->block_constraints.map_extent_ =
      axis->block_constraints.map_extent_ == 0 ? r->value : axis->block_constraints.map_extent_;
    axis->thread_constraints.map_extent_ =
      axis->thread_constraints.map_extent_ == 0 ? r->value : axis->thread_constraints.map_extent_;
  });
}

void GpuStrategy::InnerThreadOuterBlock() {
  if (pending_axes_.empty()) {
    return;
  }
  std::stringstream ss;
  int64_t activated_blocks = 1;
  int64_t activated_threads = 1;

  auto thread_dim = std::min(thread_limit_.size(), max_dim_);
  auto block_dim = std::min(block_limit_.size(), max_dim_);

  // tile from inner to outer and map to thread
  size_t ori_size = pending_axes_.size();
  size_t inner_dim = 0;
  for (size_t i = 0; i < ori_size; ++i) {
    TileAxis *axis;
    int64_t shape;
    std::tie(axis, shape) = pending_axes_[i];
    int64_t rest_threads = std::min(max_num_threads_ / activated_threads, thread_limit_[thread_cfg_.size()]);
    ss << "axis " << axis->index << "_" << axis->dim_axis << " shape = " << shape
       << ", rest_threads = " << rest_threads;
    auto SkipMapping = [this, &axis, &shape, &ss]() {
      if (axis->block_constraints.map_extent_ > 1) {
        pending_axes_.push_back(std::make_pair(axis, shape));
        ss << ", map to block.";
      }
      analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
    };

    if (axis->thread_constraints.map_extent_ <= 1) {
      ss << ", thread mapping is not allowed.";
      SkipMapping();
      continue;
    }

    if (rest_threads <= 1 || thread_cfg_.size() >= thread_dim || inner_dim >= max_dim_) {
      ss << ", no thread/dim rests";
      // finish mapping threads, tile rest to minimum for block mapping
      axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), TileLevel::LEVEL1);
      SkipMapping();
      continue;
    }

    ++inner_dim;
    auto use = GetThreadSize(rest_threads, shape);
    activated_threads *= use;
    ss << ", use = " << use << ", actived threads = " << activated_threads;
    analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
    thread_cfg_.emplace_back(use);
    pending_axes_.push_back(std::make_pair(axis, std::max<int64_t>(ceil(static_cast<float>(shape) / use), 1)));
    axis->TileRestrainToSingleValue(CastIntToExpr(use), TileLevel::LEVEL1);
  }

  std::vector<size_t> indexing;
  for (size_t i = 0; i < block_dim; ++i) {
    block_cfg_.emplace_back(1);
  }
  // If all axes for block mapping are element-wise, we can map them in any order
  // so we need a greedy algorithm to map the most blocks;
  // otherwise, we can simply map from outer to inner in sequence.
  bool is_pure_elem = true;
  for (size_t i = pending_axes_.size() - 1; i >= ori_size; --i) {
    is_pure_elem = is_pure_elem && IsElemWiseAxis(pending_axes_[i].first);
  }
  if (is_pure_elem) {
    std::map<int64_t, std::vector<size_t>, std::greater<int64_t>> sorted_by_gcd;
    for (size_t i = pending_axes_.size() - 1; i >= ori_size; --i) {
      auto use = (max_num_blocks_ > 0 && pending_axes_[i].second > 0)
                   ? TilingAnalyzer::FindDivisibleTilingFactor(max_num_blocks_, pending_axes_[i].second)
                   : 1;
      if (sorted_by_gcd.find(use) == sorted_by_gcd.end()) {
        sorted_by_gcd[use] = {i};
      } else {
        sorted_by_gcd[use].emplace_back(i);
      }
    }

    for (const auto &it : sorted_by_gcd) {
      auto index_list = it.second;
      for (const auto &i : index_list) {
        if (pending_axes_.size() - i > block_dim) {
          auto axis = pending_axes_[i].first;
          ss << "axis " << axis->index << "_" << axis->dim_axis
             << " exceeded block dim and should be mapped to block for higher performance, consider flatten";
          analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
          continue;
        }
        indexing.emplace_back(i);
      }
    }
  } else {
    for (size_t i = pending_axes_.size() - 1; i >= ori_size; --i) {
      indexing.emplace_back(i);
    }
  }

  // map outer band to block according to predefined indice
  for (const auto &i : indexing) {
    TileAxis *axis;
    int64_t shape;
    std::tie(axis, shape) = pending_axes_[i];
    auto rest_blocks = std::min(max_num_blocks_ / activated_blocks, block_limit_[pending_axes_.size() - 1 - i]);
    rest_blocks = std::min(rest_blocks, axis->block_constraints.map_extent_);
    ss << "axis " << axis->index << "_" << axis->dim_axis << " shape = " << shape << ", rest blocks = " << rest_blocks;
    if (block_count_ >= static_cast<int>(block_dim)) {
      ss << "-> No mapping.";
      analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
      continue;
    }
    auto use = (rest_blocks > 0 && shape > 0) ? TilingAnalyzer::FindDivisibleTilingFactor(rest_blocks, shape) : 1;
    activated_blocks *= use;
    ss << ", use = " << use << ", actived blocks = " << activated_blocks;
    analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
    block_cfg_[pending_axes_.size() - 1 - i] = use;
    if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib() || axis->mc_sup) {
      ++block_count_;
    }
    auto extent = axis->range_extent.as<IntImm>()->value;
    auto inner_extent = axis->l1_constraints.tile_extent_.as<IntImm>()->value;
    axis->l1_constraints.tile_extent_ =
      std::min(inner_extent, std::max<int64_t>(ceil(static_cast<float>(extent) / use), 1));
  }
}

void GpuStrategy::SetMappingConfig() {
  std::stringstream ss;
  if (thread_cfg_.empty()) {
    thread_cfg_.emplace_back(1);
  }
  if (block_cfg_.empty()) {
    block_cfg_.emplace_back(1);
  }
  std::string block_str = block_count_ == 0 ? "1" : "";
  for (int i = block_cfg_.size() - 1; i >= 0; --i) {
    if (i >= block_count_) {
      continue;
    }
    block_str += (std::to_string(block_cfg_[i]) + " ");
  }
  bool reverse_binding = (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib() &&
                          analyzer_->scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION);
  std::string thread_str = "";
  if (reverse_binding) {
    // pad binding to at least two dim to bind reduce axis at thread y
    for (size_t i = thread_cfg_.size(); i < 2; ++i) {
      thread_cfg_.emplace_back(1);
    }
    for (int i = thread_cfg_.size() - 1; i >= 0; --i) {
      thread_str += (std::to_string(thread_cfg_[i]) + " ");
    }
  } else {
    for (const auto &size : thread_cfg_) {
      thread_str += (std::to_string(size) + " ");
    }
  }

  analyzer_->scop_info_.user_config_.SetBlockConfig(block_str);
  analyzer_->scop_info_.user_config_.SetThreadConfig(thread_str);

  ss << "Block config = " << block_str;
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
  ss << "Thread config = " << thread_str;
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
}

int64_t GpuStrategy::GetThreadSize(const int64_t rest_threads, const int64_t shape) {
  // TODO: how to set best thread size according to current rest_thread and shape
  //       is not sure and profiling test is needed.

  // Current experience is that let mapped threads divisible by warp_size to increase performance.
  if (shape > rest_threads) {
    return rest_threads;
  }
  return std::min(rest_threads, (shape - 1 + warp_sizes_) / warp_sizes_ * warp_sizes_);
}

void GpuStrategy::DetermineTemplate() {
  for (auto it : analyzer_->scop_info_.analysis_result_.GetReduceStatementMap()) {
    if (analyzer_->scop_info_.analysis_result_.GetReduceOpType(it.first) == AKG_REDUCE_AND ||
        analyzer_->scop_info_.analysis_result_.GetReduceOpType(it.first) == AKG_REDUCE_OR) {
      template_ = Template::BITWISE_REDUCTION;
      return;
    }
  }
  auto reduce_axes = analyzer_->GetAxesOfAttr("REDUCE_AXIS");
  size_t depth = 0;
  analyzer_->ForEachAxisTopDown([this, &depth](TileAxis *axis) {
    if (axis == analyzer_->RootAxis()) {
      return;
    }
    ++depth;
  });
  if (reduce_axes.size() == depth) {
    template_ = Template::ALL_REDUCE;
    return;
  }

  analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
    if (axis->range_extent.as<IntImm>() == nullptr) {
      return;
    }
    for (const auto &attr : axis->attrs) {
      for (const auto &e : excluded_attr_) {
        if (attr.attr_key.find(e) != std::string::npos) {
          if (e == "REDUCE" && template_ < Template::REDUCTION) {
            template_ = Template::REDUCTION;
          }
          if (e == "TRANSPOSE" && template_ < Template::TRANSPOSE) {
            template_ = Template::TRANSPOSE;
          }
        }
      }
    }
  });
  if (template_ < Template::PURE_ELEM) {
    template_ = Template::PURE_ELEM;
  }
}

void GpuStrategy::AdjustThreadMappingLimit() {
  std::stringstream ss;
  std::vector<int64_t> map_mins;
  ss << "Original thread limit = ";
  for (auto tl : thread_limit_) {
    ss << tl << ", ";
  }
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
  analyzer_->ForEachAxisTopDown([this, &map_mins](TileAxis *axis) {
    if (axis == this->analyzer_->RootAxis()) {
      return;
    }
    map_mins.emplace_back(axis->thread_constraints.map_min_);
  });
  std::reverse(map_mins.begin(), map_mins.end());
  auto map_size = thread_limit_.size();
  for (size_t i = 0; i < map_mins.size(); ++i) {
    if (i > map_size) {
      continue;
    }
    for (size_t j = 0; j < map_size; ++j) {
      if (j == i) {
        continue;
      }
      int64_t res = floor(static_cast<float>(thread_limit_[j]) / map_mins[i]);
      thread_limit_[j] = res;
    }
  }
  ss << "Adjust thread limit by axes' mapping mins = ";
  for (auto tl : thread_limit_) {
    ss << tl << ", ";
  }
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
}

bool GpuStrategy::IsElemWiseAxis(TileAxis *axis) {
  if (axis->range_extent.as<IntImm>() == nullptr) {
    return false;
  }
  for (const auto &attr : axis->attrs) {
    for (const auto &e : excluded_attr_) {
      if (attr.attr_key.find(e) != std::string::npos) {
        return false;
      }
    }
  }
  return true;
}

// No constraint found in cuda

void ModStrategy::AddGpuConstraint() {}

void CastStrategy::AddGpuConstraint() {}

void CustomTilingStrategy::AddGpuConstraint() {}

void ConflictTreeRangeStrategy::AddGpuConstraint() {}

void VectorizedStrategy::AddGpuConstraint() {}

void DmaAlignStrategy::AddGpuConstraint() {}

void TensorOfTensorStrategy::AddGpuConstraint() {}

void PassDownAttrStrategy::AddGpuConstraint() {}

void DynamicShapeLimitStrategy::AddGpuConstraint() {}

void DynamicBoundStrategy::AddGpuConstraint() {}

void ShiftAxisStrategy::AddGpuConstraint() {}

void ModShiftAxisStrategy::AddGpuConstraint() {}

void ConvStrategy::AddGpuConstraint() {}

void GemmStrategy::AddGpuConstraint() {}

// end of null constraint

}  // namespace poly
}  // namespace ir
}  // namespace akg
