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

#include "register_memory_manager.h"
#include "poly/scop.h"
#include "poly/dma_inject.h"
#include "poly/schedule_tree_util.h"

namespace akg {
namespace ir {
namespace poly {

isl::union_set RegisterMemoryManager::GatherMappingsTo(MappingCfg *cfg) {
  isl::schedule_node root = schedule_.get_root();
  auto domain_node = root.as<isl::schedule_node_domain>();
  auto domain = domain_node.domain();
  auto mapping_filters = CollectNode<isl::schedule_node_filter>(schedule_);

  std::vector<isl::id> filters;
  for (size_t idx = 0; idx < cfg->bound; ++idx) {
    auto value = cfg->GetAt(idx);
    auto id = isl::id(root.ctx(), value.first);
    filters.push_back(id);
  }
  mapping_filters = FilterNode(mapping_filters, filters);

  auto mapping = isl::union_set::empty(domain.ctx());
  for (auto item : mapping_filters) {
    if (item.isa<isl::schedule_node_filter>()) {
      auto filter = item.as<isl::schedule_node_filter>();
      if (filter.has_parent() && !filter.parent().isa<isl::schedule_node_mark>()) {
        continue;
      }

      isl::union_set uset = filter.get_filter();
      std::vector<isl::set> vset;
      uset.foreach_set([&vset](isl::set s) { vset.push_back(s); });
      if (!vset.empty()) {
        auto filter_name = vset[0].get_tuple_name();
        if (filter_name == READ_ID_NAME || filter_name == WRITE_ID_NAME) {
          continue;
        }
      }

      mapping = mapping.unite(filter.filter());
    }
  }
  return mapping;
}

isl::schedule RegisterMemoryManager::HoistRegisterMemoryOnDepth(isl::schedule_node &node, size_t depth) {
  auto block_cfg = scop_info_.user_config_.GetBlockConfig();
  auto res_node = node;
  auto block_mapping = GatherMappingsTo(block_cfg);
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  auto mapping = GatherMappingsTo(thread_cfg).intersect(block_mapping);

  auto partial_sched = LocalSchedule(node);
  auto tmp_sched = partial_sched.intersect_domain(mapping);
  CreateTensorCluster(node, tmp_sched);

  isl::schedule_node root_node = schedule_.get_root();

  isl::schedule sch = schedule_;
  if (memory_exceeding_) {
    depth = depth + 1;
    sch = HoistRegisterMemory(root_node, depth);
    return sch;
  } else {
    auto thread_schedule = MapDomainAllWithType(root_node, thread_cfg, scop_info_.upa_node_mapping_, THREAD_MARKER);
    auto block_schedule = MapDomainAllWithType(root_node, block_cfg, scop_info_.upa_node_mapping_, BLOCK_MARKER);

    auto tmp_node = res_node;
    if (node.isa<isl::schedule_node_band>()) {
      tmp_node = res_node.child(0);
    }

    auto partial_sched_mupa = ShortScheduleMupa(root_node, tmp_node);
    auto partial_sched_with_block = isl::union_map::from(partial_sched_mupa).intersect_domain(block_mapping);
    partial_sched_mupa = partial_sched_mupa.flat_range_product(block_schedule);
    for (size_t index = 0; index < scop_info_.analysis_result_.buffer_def_infos_.size(); index++) {
      BufferDefInfo &buffer_info = scop_info_.analysis_result_.buffer_def_infos_[index];

      if (buffer_info.dst_tensor_id.to_str().find("shared") != std::string::npos) {
        continue;
      }

      auto fp_cluster = buffer_info.GetFootPrintClusterGPU(res_node);

      if (fp_cluster == nullptr || !fp_cluster->foot_print_.box.is_valid()) {
        continue;
      }

      auto tensor_id = buffer_info.tensor_id;
      auto box_sizes = fp_cluster->GetFixedBoxSizes();

      if (box_sizes.size() == 0) {
        LOG(FATAL) << "Can not manage a scalar tensor in register memory promotion";
      }

      partial_sched_mupa = partial_sched_mupa.flat_range_product(thread_schedule);

      if (!IsPromote(*fp_cluster, partial_sched_mupa, thread_schedule)) {
        continue;
      }

      if (!ReuseTensorCluster(*fp_cluster, partial_sched_mupa)) {
        continue;
      }

      auto active_domains = CollectDomain(res_node);
      isl::id dst_tensor_id = GpuDstId(GpuMemType::LOCAL, tensor_id);
      GatherBufferFootprintDefInfo(res_node, buffer_info);
      node = PlaceOuterDataCopyBelow(scop_info_, node, *fp_cluster, tensor_id, dst_tensor_id, partial_sched,
                                     schedule_.get_domain().get_space());

      // active_buffer_footprints for codegen
      auto dst_id = GpuDstId(GpuMemType::LOCAL, tensor_id);
      scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(std::make_pair(
        active_domains,
        BufferedFootPrintInfo{std::shared_ptr<TensorFootprintCluster>(std::move(fp_cluster)), partial_sched, dst_id}));
      buffer_info.find_buffer = true;
    }
    sch = node.get_schedule();
    return sch;
  }
}

bool RegisterMemoryManager::UnrolledLoop(const TensorFootprintCluster &fp_cluster) {
  auto box_sizes = fp_cluster.GetFixedBoxSizes();
  size_t tmp_size = 1;
  for (auto size : box_sizes) {
    tmp_size = tmp_size * size;
  }
  if (tmp_size != 1) {
    return true;
  }
  return false;
}

bool RegisterMemoryManager::IsPromote(const TensorFootprintCluster &fp_cluster,
                                      const isl::multi_union_pw_aff &partial_sched_mupa,
                                      const isl::multi_union_pw_aff &thread_schedule) {
  auto original_access = fp_cluster.OrigianlAccessRelations();
  auto map = isl::union_map::from(partial_sched_mupa);
  map = map.range_product(original_access);
  map = map.apply_domain(isl::union_map::from(thread_schedule));
  return map.is_injective();
}

bool RegisterMemoryManager::ReuseTensorCluster(const TensorFootprintCluster &cluster,
                                               const isl::multi_union_pw_aff &outer_pw_aff) {
  isl::union_map out_schedule = isl::union_map::from(outer_pw_aff);
  out_schedule = out_schedule.range_product(cluster.OrigianlAccessRelations());
  return !out_schedule.is_injective();
}

void RegisterMemoryManager::CreateTensorCluster(const isl::schedule_node &node, const isl::union_map &outer_sch) {
  isl::union_map reads = scop_info_.analysis_result_.GetReads();
  isl::union_map writes = scop_info_.analysis_result_.GetWrites();
  isl::union_map copyin = scop_info_.analysis_result_.GetCopyin();
  isl::union_map fake_copyin = scop_info_.analysis_result_.GetFakeCopyin();

  auto read_map = scop_info_.StmtReadMap();
  auto write_map = scop_info_.StmtWriteMap();
  auto stmt_map = scop_info_.analysis_result_.GetStmtOpInfoMap();
  std::vector<isl::id> tensor_list;
  std::unordered_set<isl::id, isl::IslIdIslHash> id_sets;
  for (auto item : read_map) {
    for (auto item_id : item.second) {
      id_sets.insert(item_id);
    }
  }
  for (auto item : write_map) {
    for (auto item_id : item.second) {
      id_sets.insert(item_id);
    }
  }

  std::set<std::string> shared_tensor_ids;
  for (auto buffer : scop_info_.analysis_result_.buffer_def_infos_) {
    shared_tensor_ids.insert(buffer.tensor_id.get_name());
  }
  if (!configed_tensors_.empty()) {
    std::set<std::string> tensor_sets;
    for (const auto &item : configed_tensors_) {
      if (tensor_sets.count(item) == 0) {
        tensor_sets.emplace(item);
      }
    }
    id_sets.clear();
    for (auto item : tensor_sets) {
      id_sets.insert(isl::id(scop_info_.ctx_, item));
    }
  }

  for (auto item : id_sets) {
    if (!shared_tensor_ids.count(item.get_name())) {
      tensor_list.push_back(item);
    }
  }

  std::vector<BufferDefInfo> promoted_infos;

  for (const auto &item : tensor_list) {
    isl::id dst_tensor_id = GpuDstId(GpuMemType::LOCAL, item);
    std::vector<size_t> buffer_sizes;
    std::vector<std::pair<isl::id, MemType>> data_stream;
    data_stream.push_back(std::make_pair(item, MemType::DDR));
    data_stream.push_back(std::make_pair(item, MemType::LOCAL_));
    BufferDefInfo promoted_info = BufferDefInfo{item,
                                                dst_tensor_id,
                                                item,
                                                MemType::DDR,
                                                "",
                                                false,
                                                false,
                                                data_stream,
                                                Tensor(),
                                                Handle(),
                                                buffer_sizes,
                                                nullptr,
                                                isl::union_map::empty(isl::space(scop_info_.ctx_, 0))};
    promoted_info.footprints_cluster =
      TensorFootprintCluster::HoistBufferFootprintCluster(outer_sch, item, reads, copyin, writes, fake_copyin);
    if (promoted_info.footprints_cluster != nullptr) {
      promoted_info.footprint_cluster_map.emplace_back(std::make_pair(node, promoted_info.footprints_cluster));
      promoted_infos.push_back(promoted_info);
    }
  }

  IsOutofMemory(promoted_infos);

  if (!memory_exceeding_) {
    for (auto promoted_info : promoted_infos) {
      scop_info_.analysis_result_.buffer_def_infos_.push_back(promoted_info);
    }
  }
}

void RegisterMemoryManager::IsOutofMemory(std::vector<BufferDefInfo> promoted_infos) {
  memory_exceeding_ = false;
  for (auto promoted_info : promoted_infos) {
    auto box_sizes = promoted_info.footprints_cluster->GetFixedBoxSizes();
    if (!box_sizes.empty()) {
      auto tensor_size = box_sizes[0];
      for (unsigned int i = 1; i < box_sizes.size(); ++i) {
        tensor_size = tensor_size * box_sizes[i];
      }

      if (tensor_size > MAX_REGISTER_TENSOR_SIZE) {
        memory_exceeding_ = true;
        break;
      }
    }
  }
}

void RegisterMemoryManager::GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info) {
  auto fp_cluster = tensor_info.footprints_cluster;
  std::vector<size_t> sizes;
  if (fp_cluster == nullptr) {
    tensor_info.AddSize(node, sizes);
    return;
  }
  sizes = fp_cluster->GetFixedBoxSizes();

  isl::id tensor_id = tensor_info.tensor_id;
  isl::id cluster_id = tensor_info.dst_tensor_id;

  // build a Halide Node for cluster_id
  Array<Expr> shapes;
  for (auto i : sizes) {
    shapes.push_back(Expr(static_cast<int>(i)));
  }

  Type type = scop_info_.GetDtypeOf(tensor_id);
  Tensor tensor = placeholder(shapes, type, cluster_id.get_name());
  const Buffer buffer = decl_buffer(shapes, scop_info_.GetDtypeOf(tensor_id), cluster_id.get_name());
  scop_info_.user_config_.SetBind(tensor, buffer);

  tensor_info.sizes = sizes;
  tensor_info.tensor = tensor;
  tensor_info.data_type = type;
  tensor_info.AddSize(node, sizes);
}

size_t RegisterMemoryManager::UpdateDepth(const isl::schedule_node &node) {
  auto band = node.as<isl::schedule_node_band>();
  for (size_t i = 0; i < band.n_member(); i++) {
    if (!band.member_get_coincident(i)) {
      if (i == 0) {
        return band.n_member();
      } else {
        return i;
      }
    }
  }
  return band.n_member();
}

isl::schedule RegisterMemoryManager::HoistRegisterMemory(isl::schedule_node root, size_t depth) {
  auto bands = BandsContainingScheduleDepth(root, depth);
  bands = FilterWithFunc(
    [root, depth](isl::schedule_node node) {
      auto band = node.as<isl::schedule_node_band>();
      return !IsThreadMappedMark(node) || node.schedule_depth() + band.n_member() == depth;
    },
    bands);
  bands = BandsSplitAfterDepth(bands, root, depth);

  isl::schedule tmp_sch;
  for (auto band : bands) {
    if (IsThreadMappedMark(band)) {
      band = band.child(0);
    }
    tmp_sch = HoistRegisterMemoryOnDepth(band, depth);
  }
  return tmp_sch;
}

isl::schedule RegisterMemoryManager::Run(isl::schedule sch) {
  LOG(INFO) << ">>>>>>>>Register memory promotion<<<<<<<<<<<<<<<";
  schedule_ = sch;
  auto root = sch.get_root();

  auto res_node = GetOuterBand(root);
  if (res_node.isa<isl::schedule_node_band>()) {
    auto depth = UpdateDepth(res_node);
    if (scop_info_.user_config_.GetRegisterDepth() >= 0) {
      depth = scop_info_.user_config_.GetRegisterDepth();
    }
    sch = HoistRegisterMemory(root, depth);
  }
  return sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg