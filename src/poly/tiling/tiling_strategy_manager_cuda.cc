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

void GpuDmaAnalysisStrategy::AddGpuConstraint() {
  analyzer_->ForEachAxisTopDown(
    [](TileAxis *axis) { axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), TileLevel::LEVEL1); });
}

void CastStrategy::AddGpuConstraint() { MarkDataSize(); }

void ReduceStrategy::AddGpuConstraint() {
  // TODO: compare XLA's reduction tiling/mapping strategy with current strategy
  reduce_axes_ = analyzer_->GetAxesOfAttr(AT_REDUCE_AXIS);
  size_t depth = 0;
  analyzer_->ForEachAxisTopDown([this, &depth](TileAxis *axis) {
    if (!has_transpose_) {
      for (const auto &attr : axis->attrs) {
        if (attr.attr_key.find(AT_TRANSPOSE) != std::string::npos) {
          has_transpose_ = true;
          break;
        }
      }
    }

    if (axis == analyzer_->RootAxis()) {
      return;
    }
    ++depth;
    if (axis->mc_sup) {
      injective_axes_.emplace_back(axis);
      return;
    }
    if (std::count(reduce_axes_.begin(), reduce_axes_.end(), axis)) {
      return;
    }
    reduce_axes_.emplace_back(axis);
  });
  all_reduce_ = reduce_axes_.size() == depth;
  if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
    AkgReduceLibStrategyOnGpu();
  } else {
    SimpleStrategyOnGpu();
  }
}

void ReduceStrategy::SimpleStrategyOnGpu() {
  if (all_reduce_ || has_transpose_) {
    auto extent = all_reduce_ ? MIN_TILE : warp_sizes_;
    bool is_tuning = analyzer_->scop_info_.user_config_.GetIsTuning();
    for (auto axis : reduce_axes_) {
      axis->block_constraints.map_extent_ = MIN_TILE;
      axis->thread_constraints.map_extent_ = MIN_TILE;
      if (!is_tuning) {
        axis->TileRestrainToSingleValue(CastIntToExpr(extent), TileLevel::LEVEL1);
      }
    }
  }
}

void ReduceStrategy::AkgReduceLibStrategyOnGpu() {
  // disable atomic-add for bitwise-reduction
  bool disable_atomic = !analyzer_->scop_info_.user_config_.GetEnableAtomicAdd();
  if (!disable_atomic) {
    for (auto it : analyzer_->scop_info_.analysis_result_.GetReduceStatementMap()) {
      if (analyzer_->scop_info_.analysis_result_.GetReduceOpType(it.first) == AKG_REDUCE_AND ||
          analyzer_->scop_info_.analysis_result_.GetReduceOpType(it.first) == AKG_REDUCE_OR) {
        disable_atomic = true;
        break;
      }
    }
  }
  if (disable_atomic) {
    for (auto axis : reduce_axes_) {
      axis->block_constraints.map_extent_ = MIN_TILE;
    }
  }

  // disable atomic-add for post reduce tensors
  DealWithPostReduceTensors();

  if (has_transpose_) {
    for (auto axis : reduce_axes_) {
      axis->TileRestrainEntire(TileLevel::LEVEL1);
      axis->block_constraints.map_extent_ = MIN_TILE;
    }
  }

  bool square_thread = analyzer_->scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION;
  int64_t total_reduce_size = 1;
  int64_t total_injective_size = 1;
  int64_t injective_threads = 1;
  int64_t reduce_threads = 1;
  int64_t possible_reduce_blocks = 1;
  int64_t possible_injective_blocks = 1;

  if (!all_reduce_) {
    DealWith4DFusedReduce();
  }
  bool use_local = UseRegisterMem();

  for (auto axis : reduce_axes_) {
    CHECK(axis->range_extent.as<IntImm>());
    total_reduce_size *= axis->range_extent.as<IntImm>()->value;
    if (axis->block_constraints.map_extent_ == 0) {
      possible_reduce_blocks *= axis->range_extent.as<IntImm>()->value;
    } else {
      possible_reduce_blocks *= axis->block_constraints.map_extent_;
    }
    if (axis->thread_constraints.map_min_ == axis->thread_constraints.map_extent_ &&
        axis->thread_constraints.map_extent_ != 0) {
      reduce_threads *= axis->thread_constraints.map_min_;
    }
  }
  for (auto axis : injective_axes_) {
    CHECK(axis->range_extent.as<IntImm>());
    total_injective_size *= axis->range_extent.as<IntImm>()->value;
    if (axis->block_constraints.map_extent_ == 0) {
      possible_injective_blocks *= axis->range_extent.as<IntImm>()->value;
    } else {
      possible_injective_blocks *= axis->block_constraints.map_extent_;
    }
    if (axis->thread_constraints.map_min_ == axis->thread_constraints.map_extent_ &&
        axis->thread_constraints.map_extent_ != 0) {
      injective_threads *= axis->thread_constraints.map_min_;
    }
  }
  bool is_special_4d = reduce_threads != 1 || injective_threads != 1;
  if (is_special_4d) {
    return;
  }

  int64_t min_blocks = square_thread ? 32 : 512;
  int64_t min_elem_per_thread = use_local ? 2 : 8;

  if (total_injective_size * total_reduce_size / min_blocks / max_num_threads_ < min_elem_per_thread) {
    square_thread = true;
    min_blocks = 32;
  }

  std::pair<int64_t, int64_t> tx_range{1, max_num_threads_};
  std::pair<int64_t, int64_t> ty_range{1, max_num_threads_};
  if (square_thread) {
    tx_range.first = std::min(warp_sizes_, total_injective_size);
    ty_range.first = std::min<int64_t>(8, total_reduce_size);
    tx_range.second = std::min<int64_t>(tx_range.second, ceil(static_cast<float>(tx_range.second) / ty_range.first));
    tx_range.second = std::min(tx_range.second, total_injective_size);
  } else {
    tx_range.first = std::min(warp_sizes_, total_reduce_size);
    ty_range.first = std::min<int64_t>(8, total_injective_size);
    tx_range.second = std::min<int64_t>(tx_range.second, ceil(static_cast<float>(tx_range.second) / ty_range.first));
    tx_range.second = std::min(tx_range.second, total_reduce_size);
  }
  ty_range.second =
    std::min(ty_range.second, static_cast<int64_t>(ceil(static_cast<float>(ty_range.second) / tx_range.second)));
  if (square_thread) {
    reduce_threads = ty_range.second;
    injective_threads = tx_range.second;
  } else {
    reduce_threads = tx_range.second;
    injective_threads = ty_range.second;
  }
  for (auto axis : reduce_axes_) {
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_MOD) {
        continue;
      }
      CHECK_NE(attr.attr_value, "");
      auto mod_value = static_cast<int>(std::strtol(attr.attr_value.c_str(), nullptr, 10));
      axis->TileRestrainMod(CastInt64ToExpr(mod_value), TileLevel::LEVEL1);
    }
    if (use_local) {
      auto tile_mod = axis->l1_constraints.tile_mod_.as<IntImm>()->value;
      while (tile_mod > reduce_threads && tile_mod % reduce_threads != 0) {
        --reduce_threads;
      }
    }
  }

  int possible_blocks =
    ceil(static_cast<float>(possible_injective_blocks * possible_reduce_blocks) / injective_threads / reduce_threads);
  int proposal = use_local ? 8 : 32;
  auto default_elem_per_thread = possible_reduce_blocks > 1
                                   ? std::max(std::min<int>(proposal, (possible_blocks / min_blocks + 1) / 2 * 2), 1)
                                   : IsHalfReduce() ? 64 : SpItemPerThread::FULL;

  auto original_ept = default_elem_per_thread;
  // try to increase thread loop (no more than twice as original)
  while (possible_blocks > default_elem_per_thread && possible_blocks % default_elem_per_thread != 0) {
    ++default_elem_per_thread;
  }
  if (original_ept * 2 < default_elem_per_thread) {
    default_elem_per_thread = original_ept;
  }
  // try to decrease thread loop (no less than half of original)
  while (possible_blocks > default_elem_per_thread && possible_blocks % default_elem_per_thread != 0) {
    --default_elem_per_thread;
  }
  if (default_elem_per_thread * 2 < original_ept) {
    default_elem_per_thread = original_ept;
  }
  std::stringstream ss;
  ss << "total_injective_size " << total_injective_size << " total_reduce_size " << total_reduce_size;
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);

  ss << "injective_threads " << injective_threads << " reduce_threads " << reduce_threads;
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);

  ss << "possible_blocks " << possible_blocks << " possible_injective_blocks " << possible_injective_blocks
     << " possible_reduce_blocks " << possible_reduce_blocks << " default_elem_per_thread " << default_elem_per_thread;
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);

  ss << "tx:[" << tx_range.first << ", " << tx_range.second << "]; ty:[" << ty_range.first << ", " << ty_range.second
     << "]";
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);

  for (auto axis : injective_axes_) {
    axis->thread_constraints.map_min_ = injective_threads;
    axis->thread_constraints.map_extent_ = injective_threads;
    axis->thread_constraints.item_process_ = MIN_TILE;
  }
  for (auto axis : reduce_axes_) {
    axis->thread_constraints.map_extent_ = reduce_threads;
    axis->thread_constraints.item_process_ = default_elem_per_thread;
  }
}

bool ReduceStrategy::UseRegisterMem() {
  for (auto &it : analyzer_->buf_info_) {
    auto buf = it.second.get();
    CHECK(buf);
    if (buf->scope == TilingMemScope::MEM_SCOPE_LOCAL) {
      return true;
    }
  }
  return false;
}

bool ReduceStrategy::IsHalfReduce() {
  for (const auto axis : reduce_axes_) {
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_REDUCE_AXIS) {
        continue;
      }
      auto red_tensor_name = attr.attr_value;
      auto it = axis->data_size.find(red_tensor_name);
      if (it != axis->data_size.end() && *std::min_element(it->second.begin(), it->second.end()) == 2) {
        return true;
      }
    }
  }
  return false;
}

void ReduceStrategy::DealWith4DFusedReduce() {
  auto mod_axes = analyzer_->GetAxesOfAttr(AT_MOD);
  for (auto axis : mod_axes) {
    if (axis->HasAttr(AT_REDUCE_AXIS) || axis->mc_sup == 0) {
      continue;
    }
    int last_mod_value = -1;
    size_t num_mod_axis = 0;
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_MOD) {
        continue;
      }
      CHECK_NE(attr.attr_value, "");
      last_mod_value = static_cast<int>(std::strtol(attr.attr_value.c_str(), nullptr, 10));
      ++num_mod_axis;
    }
    if (num_mod_axis < 1) {
      continue;
    }
    axis->TileRestrainToSingleValue(CastIntToExpr(last_mod_value), TileLevel::LEVEL1);
    if (last_mod_value > max_num_threads_) {
      LOG(WARNING) << "Cannot bind axis to " << last_mod_value << " threads, maximal thread number is "
                   << max_num_threads_
                   << ". If fusing more than two axes together, footprint box calculated by isl may not be correct.";
      continue;
    }
    axis->thread_constraints.map_extent_ = last_mod_value;
  }
}

void ReduceStrategy::DealWithPostReduceTensors() {
  std::unordered_set<std::string> post_reduce_tensors;
  auto root = analyzer_->RootAxis();
  for (const auto &attr : root->attrs) {
    if (attr.attr_key != AT_POST_FUSION_REDUCE_TENSOR) {
      continue;
    }
    auto tensor_name = attr.attr_value;
    post_reduce_tensors.insert(tensor_name);
  }

  for (const auto axis : reduce_axes_) {
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_REDUCE_AXIS) {
        continue;
      }
      auto red_tensor_name = attr.attr_value;
      if (post_reduce_tensors.find(red_tensor_name) == post_reduce_tensors.end()) {
        continue;
      }
      axis->block_constraints.map_min_ = MIN_TILE;
      axis->block_constraints.map_extent_ = MIN_TILE;
      axis->thread_constraints.item_process_ = SpItemPerThread::FULL;
    }
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
  max_num_threads_ = analyzer_->scop_info_.user_config_.GetMaxElemPerThread();
  DetermineTemplate();
  std::stringstream ss;
  ss << "Use template " << template_map_[template_];
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
  if (template_ == Template::CUSTOM_CONFIG) {
    auto thread_config = analyzer_->scop_info_.user_config_.GetThreadConfig();
    for (size_t i = 0; i < thread_config->bound; ++i) {
      thread_limit_.emplace_back(thread_config->GetAt(i).second);
    }
  } else if (template_ == Template::REDUCTION || template_ == Template::BITWISE_REDUCTION) {
    thread_limit_ = {max_num_threads_, max_num_threads_};
  } else if (template_ == Template::ALL_REDUCE) {
    if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
      thread_limit_ = {max_num_threads_, max_num_threads_};
    } else {
      thread_limit_ = {1};
    }
  } else if (template_ == Template::TRANSPOSE_OP) {
    analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
      axis->thread_constraints.item_process_ = std::max(axis->thread_constraints.item_process_, min_elem_for_io_bound_);
    });
    thread_limit_ = {max_num_threads_, max_num_threads_};
  } else if (template_ == Template::MATMUL) {
    // This is a naive tiling strategy used in gpu when thread and block configs are already set.
    // This strategy will tile up to three inner-most axes to 32 (for thread binding).
    thread_limit_ = {32, 8};
  } else {
    thread_limit_ = {max_num_threads_, max_num_threads_, max_num_threads_};
  }

  if (template_ != Template::CUSTOM_CONFIG) {
    AdjustThreadMappingLimit();
  }

  if (template_ == Template::CUSTOM_CONFIG) {
    auto block_config = analyzer_->scop_info_.user_config_.GetBlockConfig();
    for (int i = block_config->bound - 1; i >= 0; --i) {
      block_limit_.emplace_back(block_config->GetAt(i).second);
    }
  } else if (template_ <= Template::REDUCTION) {
    block_limit_ = {max_num_blocks_, max_num_blocks_, max_num_blocks_};
  } else if (template_ == Template::ALL_REDUCE) {
    if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
      block_limit_ = {max_num_blocks_, max_num_blocks_, max_num_blocks_};
    } else {
      block_limit_ = {1};
    }
  } else if (template_ == Template::BITWISE_REDUCTION) {
    if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
      block_limit_ = {1};
    } else {
      block_limit_ = {max_num_blocks_, max_num_blocks_, max_num_blocks_};
    }
  } else {
    block_limit_ = {max_num_blocks_, max_num_blocks_, max_num_blocks_};
  }

  std::vector<std::string> elem_cfg = common::Split(analyzer_->scop_info_.user_config_.GetElemPerThread(), " ");
  for (size_t i = 0; i < max_dim_; ++i) {
    if (i < elem_cfg.size() && !elem_cfg[i].empty()) {
      elem_per_thread_[i] = static_cast<int64_t>(std::strtol(elem_cfg[i].c_str(), nullptr, 10));
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
    auto SkipMapping = [this, &axis, &shape, &ss, &inner_dim, &thread_dim]() {
      auto tile = inner_dim < thread_dim ? elem_per_thread_[inner_dim] : 1;
      tile = tile == SpItemPerThread::AUTO ? std::min(axis->thread_constraints.item_process_, max_elem_per_thread_)
                                           : tile == SpItemPerThread::FULL ? std::min(shape, max_elem_per_thread_) : 1;
      if (axis->block_constraints.map_extent_ > 1) {
        tile =
          std::max(tile, std::max<int64_t>(ceil(static_cast<float>(shape) / axis->block_constraints.map_extent_), 1));
        pending_axes_.push_back(std::make_pair(axis, std::max<int64_t>(ceil(static_cast<float>(shape) / tile), 1)));
        ss << ", map to block.";
      } else {
        tile = std::min(tile, shape);
      }
      axis->TileRestrainLower(tile, TileLevel::LEVEL1);
      ss << ", tile = " << tile;
      analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
    };

    if (template_ != Template::CUSTOM_CONFIG) {
      rest_threads = std::min(rest_threads, axis->thread_constraints.map_extent_);
    }

    if (rest_threads <= 1 || thread_cfg_.size() >= thread_dim || inner_dim >= max_dim_) {
      ss << ", no thread/dim rests";
      SkipMapping();
      continue;
    }
    auto item = elem_per_thread_[inner_dim] == SpItemPerThread::AUTO ? axis->thread_constraints.item_process_
                                                                     : elem_per_thread_[inner_dim];
    item = std::min(item, max_elem_per_thread_);
    auto use = GetThreadSize(rest_threads, inner_dim, shape, item);
    activated_threads *= use;
    ss << ", use = " << use << ", activated threads = " << activated_threads;
    thread_cfg_.emplace_back(use);
    auto tile = TileAfterThreadMapping(axis, inner_dim, use, item);
    pending_axes_.push_back(std::make_pair(axis, std::max<int64_t>(ceil(static_cast<float>(shape) / tile), 1)));
    analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
    ++inner_dim;
  }

  std::vector<size_t> indexing;
  for (size_t i = 0; i < block_dim; ++i) {
    block_cfg_.emplace_back(1);
  }
  // If all axes for block mapping are element-wise, we can map them in any order
  // so we need a greedy algorithm to map the most blocks;
  // otherwise, we can simply map from outer to inner in sequence.
  if (template_ == Template::PURE_ELEM) {
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
    ss << ", use = " << use << ", activated blocks = " << activated_blocks;
    block_cfg_[pending_axes_.size() - 1 - i] = use;
    if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib() || axis->mc_sup) {
      ++block_count_;
    }
    CHECK(axis->range_extent.as<IntImm>());
    auto extent = axis->range_extent.as<IntImm>()->value;
    axis->TileRestrainUpper(std::max<int64_t>(ceil(static_cast<float>(extent) / use), 1), TileLevel::LEVEL1);
    ss << ", tile range = [" << axis->l1_constraints.tile_min_ << ", " << axis->l1_constraints.tile_extent_ << "]";
    analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
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

int64_t GpuStrategy::GetThreadSize(const int64_t rest_threads, size_t inner_dim, const int64_t shape,
                                   const int64_t item) {
  // TODO: how to set best thread size according to current rest_thread and shape
  //       is not sure and profiling test is needed.

  // Current experience is that let mapped threads divisible by warp_size to increase performance.
  int64_t thread_extent = item == SpItemPerThread::FULL ? rest_threads : ceil(static_cast<float>(shape) / item);
  if (thread_extent > rest_threads || template_ == Template::CUSTOM_CONFIG) {
    return rest_threads;
  }
  auto proposal = inner_dim == 0 ? ((thread_extent - 1 + warp_sizes_) / warp_sizes_ * warp_sizes_) : thread_extent;
  return std::min(rest_threads, proposal);
}

int64_t GpuStrategy::TileAfterThreadMapping(TileAxis *axis, size_t inner_dim, int64_t thread_size, const int64_t item) {
  std::stringstream ss;
  auto shape = axis->range_extent.as<IntImm>()->value;
  auto tile_min = axis->l1_constraints.tile_min_.as<IntImm>()->value;
  auto tile_mod = axis->l1_constraints.tile_mod_.as<IntImm>()->value;
  auto tile_extent = axis->l1_constraints.tile_extent_.as<IntImm>()->value;
  if (tile_min == tile_extent && tile_extent != MIN_TILE) {
    ss << "tile extent is already determined = " << tile_extent;
    analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
    return tile_extent;
  }

  auto tile = item == SpItemPerThread::FULL ? std::min(tile_extent, thread_size * max_elem_per_thread_)
                                            : std::min(tile_extent, thread_size * item);
  if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
    if (tile < tile_mod) {
      // tile axis with mod value
      // e.g. tile cc0 with 128 in the following code
      // for cc0 in 1024:
      //    A[0, floormod(cc0, 256)] = B[floordiv(cc0, 256), floormod(cc0, 256)]
      while (tile_mod % tile != 0 && tile > thread_size) {
        --tile;
      }
    } else {
      // tile axis with div value
      // e.g. tile cc0 with 512 in the following code (which equals tile floordiv(cc0, 256) with 2)
      // for cc0 in 1024:
      //    A[0, floormod(cc0, 256)] = B[floordiv(cc0, 256), floormod(cc0, 256)]
      while (shape % tile != 0 && tile > thread_size) {
        --tile;
      }
    }
  }
  if (template_ == Template::CUSTOM_CONFIG && tile < thread_size) {
    tile = thread_size;
    ss << "use custom config, tile = thread size";
  }
  ss << "axis " << axis->index << "_" << axis->dim_axis << " elem_per_thread = " << item << ", tile = " << tile;
  analyzer_->logger_.AppendLog(GPU_MAPPING, ss);
  axis->TileRestrainToSingleValue(CastInt64ToExpr(tile), TileLevel::LEVEL1);
  return tile;
}

void GpuStrategy::DetermineTemplate() {
  if (analyzer_->scop_info_.user_config_.GetThreadConfig() != nullptr &&
      analyzer_->scop_info_.user_config_.GetBlockConfig() != nullptr &&
      analyzer_->scop_info_.user_config_.GetThreadConfig()->bound > 0 &&
      analyzer_->scop_info_.user_config_.GetBlockConfig()->bound > 0) {
    template_ = Template::CUSTOM_CONFIG;
    return;
  }

  for (auto it : analyzer_->scop_info_.analysis_result_.GetReduceStatementMap()) {
    if (analyzer_->scop_info_.analysis_result_.GetReduceOpType(it.first) == AKG_REDUCE_AND ||
        analyzer_->scop_info_.analysis_result_.GetReduceOpType(it.first) == AKG_REDUCE_OR) {
      template_ = Template::BITWISE_REDUCTION;
      return;
    }
  }

  if (!analyzer_->GetAxesOfAttr(AT_GEMM).empty()) {
    template_ = Template::MATMUL;
    return;
  }

  auto reduce_axes_ = analyzer_->GetAxesOfAttr(AT_REDUCE_AXIS);

  if (reduce_axes_.empty()) {
    bool has_transpose = false;
    analyzer_->ForEachAxisTopDown([this, &has_transpose](TileAxis *axis) {
      if (has_transpose) {
        return;
      }
      for (const auto &attr : axis->attrs) {
        if (attr.attr_key.find(AT_TRANSPOSE) != std::string::npos) {
          has_transpose = true;
        }
      }
    });
    template_ = has_transpose ? Template::TRANSPOSE_OP : Template::PURE_ELEM;
    return;
  }

  size_t depth = 0;
  analyzer_->ForEachAxisTopDown([this, &depth](TileAxis *axis) {
    if (axis == analyzer_->RootAxis()) {
      return;
    }
    ++depth;
  });

  template_ = reduce_axes_.size() == depth ? Template::ALL_REDUCE : Template::REDUCTION;
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

// No constraint found in cuda

void ModStrategy::AddGpuConstraint() {}

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
