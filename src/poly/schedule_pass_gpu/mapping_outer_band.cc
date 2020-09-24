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

#include "mapping_outer_band.h"
#include "poly/schedule_tree_util.h"
#include "poly/sync_manager.h"

#include "poly/scop.h"

namespace akg {
namespace ir {
namespace poly {

isl::multi_union_pw_aff MappingOuterBand::MapDomainToWarp(const isl::schedule_node &node, MappingCfg *mapping_cfg,
                                                          isl::multi_union_pw_aff domain_threads) {
  isl::space space = isl::space(node.ctx(), 0);
  auto block_space = space.add_named_tuple_id_ui(isl::id(node.ctx(), SYNC_BLOCK), mapping_cfg->bound);
  auto bspace = block_space;
  auto warp_space = space.add_named_tuple_id_ui(isl::id(node.ctx(), SYNC_WARP), 1);

  auto block_aff = isl_aff_zero_on_domain(isl_local_space_from_space(bspace.release()));
  isl::aff aff = isl::manage(block_aff);

  auto identity = isl::multi_aff::identity(block_space.map_from_set());
  for (int i = mapping_cfg->bound - 1; i >= 0; --i) {
    auto bi = mapping_cfg->GetAt(i);
    aff = aff.scale(isl::val(node.ctx(), bi.second));
    aff = aff.add(identity.get_aff(i));
  }

  aff = aff.scale_down(isl::val(node.ctx(), WARP_SIZE)).floor();
  auto map_space = block_space.product(warp_space).unwrap();
  isl::multi_aff thread_warp = isl::multi_aff(map_space, isl::aff_list(aff));
  return domain_threads.apply(thread_warp);
}

bool MappingOuterBand::IsOuterBandWithNoCoincident(const isl::schedule_node &node) {
  int depth = node.get_tree_depth();
  isl::schedule_node ancestor_node;

  for (int i = 0; i < depth; ++i) {
    ancestor_node = node.ancestor(depth - i);
    if (auto band = ancestor_node.as<isl::schedule_node_band>()) {
      auto n_coincident = CountConsecutiveCoincident(band);
      if (band.n_member() > n_coincident) {
        return true;
      }
    }
    if (ancestor_node.isa<isl::schedule_node_sequence>()) {
      return false;
    }
  }

  return false;
}

size_t MappingOuterBand::CountConsecutiveCoincident(const isl::schedule_node_band &band_node) {
  size_t count = 0;
  while (count < band_node.n_member()) {
    if (!band_node.member_get_coincident(static_cast<int>(count))) {
      break;
    }
    ++count;
  }
  return count;
}

isl::schedule_node MappingOuterBand::FillRemainingThreads(isl::schedule_node &node, size_t begin) {
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "threadconfig is null";
  size_t end = thread_cfg->bound;
  if (begin == end) {
    return node;
  }

  CHECK(node.isa<isl::schedule_node_filter>()) << "The child of set or sequence must be a filter!";
  node = node.child(0);

  isl::union_set domain = node.get_schedule().get_domain();
  isl::space space = domain.get_space();
  space = space.set_from_params();
  isl::multi_val mv = isl::multi_val::zero(space);
  isl::multi_union_pw_aff mupa = isl::multi_union_pw_aff(domain, mv);
  node = node.insert_partial_schedule(mupa);

  isl::schedule_node_band band_node = node.as<isl::schedule_node_band>();
  auto after_map_pair = MapInnerDimToThreads(band_node, false, thread_cfg, scop_info_.upa_node_mapping_);
  auto after_map_node = after_map_pair.first;
  after_map_node = after_map_node.parent();
  return after_map_node;
}

size_t MappingOuterBand::NumMappedDescendant(const RoadMap &thread_roadmap, const isl::schedule_node parent) {
  size_t max_thread_size = 0;
  for (const auto &record : thread_roadmap) {
    auto child_node = record.first;
    auto thread_size = record.second;
    bool is_child = parent.is_equal(child_node);
    while (!is_child && child_node && child_node.has_parent()) {
      child_node = child_node.parent();
      is_child = parent.is_equal(child_node);
    }
    if (is_child) {
      max_thread_size = std::max(max_thread_size, thread_size);
    }
  }
  return max_thread_size;
}

bool MappingOuterBand::CanBeMappedToThread(const isl::schedule_node node, const RoadMap &thread_record) {
  auto band = node.as<isl::schedule_node_band>();
  if (!band || !band.permutable() || NumMappedDescendant(thread_record, node) > 0) {
    return false;
  }

  // make sure a band node in a sequence node only be mapped when all its siblings can be mapped together
  if (band.ancestor(2) && band.ancestor(2).isa<isl::schedule_node_sequence>()) {
    auto seq = band.ancestor(2).as<isl::schedule_node_sequence>();
    for (size_t i = 0; i < seq.n_children(); ++i) {
      auto filter = seq.child(i);
      if (!CanBeMappedToThread(filter.child(0), thread_record)) {
        return false;
      }
    }
  }
  return true;
}

isl::schedule MappingOuterBand::DoThreadMapping(const isl::schedule &sch) {
  auto final_schedule = sch;
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";
  if (thread_cfg->bound < 1) {
    return final_schedule;
  }

  // Step 1. Find inner-most permutable band to map threads.
  RoadMap thread_record;
  auto MapFromInner = [&thread_record, thread_cfg, this](isl::schedule_node node) -> isl::schedule_node {
    size_t num_mapped_desc = NumMappedDescendant(thread_record, node);

    if (CanBeMappedToThread(node, thread_record)) {
      auto node_bak = node;
      auto mapped_threads = MapThreadHelper(node);
      if (!IsEqual(node_bak, node)) {
        // if successfully mapped current node, we insert a map filter beyond and need to return to band node
        node = node.parent();
      }
      thread_record.emplace_back(std::make_pair(node, mapped_threads));
      return node;
    }

    // deal with band that has children mapped to threads
    if (node.n_children() > 1 && num_mapped_desc > 0) {
      for (size_t i = 0; i < node.n_children(); ++i) {
        isl::schedule_node node_child = node.child(i);
        for (const auto &record : thread_record) {
          auto child_node = record.first;
          auto thread_size = record.second;
          bool is_child = IsEqual(node_child, child_node);
          if (is_child) {
            node_child = FillRemainingThreads(node_child, thread_size);
            node = node_child.parent();
            break;
          }
        }
      }

      auto need_sync = node.isa<isl::schedule_node_sequence>();
      if (need_sync) {
        node = DoThreadSynchronization(node);
      }

      auto band = node.as<isl::schedule_node_band>();
      if (band && CountConsecutiveCoincident(band) < band.n_member()) {
        CHECK_EQ(num_mapped_desc, thread_cfg->bound) << "Must be mapped to all threads.";
        auto sync_manager = scop_info_.sync_manager_;
        sync_manager.InsertExtensionNode(band.child(0), SyncLevel::BLOCK, true);
      }
    }
    return node;
  };
  final_schedule = sch.get_root().map_descendant_bottom_up(MapFromInner).get_schedule();
  return final_schedule;
}

isl::schedule_node MappingOuterBand::DoThreadSynchronization(const isl::schedule_node &node) {
  auto sync_node = node;
  auto sync_manager = scop_info_.sync_manager_;
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";

  // Step 1. prepare info
  bool is_outer = IsOuterBandWithNoCoincident(node);
  auto domain_thread = MapDomainToThread(node, thread_cfg, scop_info_.upa_node_mapping_);
  auto domain_warp = MapDomainToWarp(node, thread_cfg, domain_thread);

  // Step 2. construct a linked list for all nodes in the input sequence node
  auto head = InitSyncLinkedList(node, domain_thread, domain_warp);

  // Step 3. Use "fewest synchronization number first" strategy to determine the
  //         optimization sync position in the sequence node.
  head = CountSyncNumberAmongLoop(head);
  auto start = GetBestSyncStartPoint(is_outer);
  auto all_syncs = DetermineOptSyncPos(head, start);
  std::sort(all_syncs.begin(), all_syncs.end(),
            [](Synchronization s1, Synchronization s2) { return s1.pos >= s2.pos; });

  // Step 4. Insert sync node (extension and filter) in the sequence node
  for (const auto &sync : all_syncs) {
    auto target = sync_node.child(sync.pos).child(0);
    sync_node = sync_manager.InsertExtensionNode(target, sync.level, true).parent().parent();
  }

  return sync_node;
}

std::vector<Synchronization> MappingOuterBand::DetermineOptSyncPos(SyncCandidate *head, int start) {
  std::vector<Synchronization> all_syncs;
  auto start_node = head->NextNCandidate(start);

  auto SplitList = [&start_node, &all_syncs](SyncLevel level) {
    if (level == SyncLevel::EMPTY) {
      return;
    }
    auto cur = start_node;
    while (cur) {
      auto opt = cur->GetOptimalSyncPos(level);
      cur = opt.first;
      bool exit = opt.second == 0;
      auto new_sync = Synchronization(level, cur->idx);
      for (const auto &old_sync : all_syncs) {
        if (new_sync.IsEqual(old_sync)) {
          exit = true;
          break;
        }
      }
      if (exit) {
        break;
      }
      all_syncs.emplace_back(new_sync);
    }
  };
  SplitList(SyncLevel::BLOCK);
  SplitList(SyncLevel::WARP);
  return all_syncs;
}

SyncCandidate *MappingOuterBand::InitSyncLinkedList(const isl::schedule_node &seq_node,
                                                    const isl::multi_union_pw_aff &domain_to_thread,
                                                    const isl::multi_union_pw_aff &domain_to_warp) {
  auto context_params = scop_info_.analysis_result_.GetContextParams();
  auto dependency = pass_info_.dependences_;
  auto seq_len = static_cast<int>(seq_node.n_children());
  auto root = std::unique_ptr<SyncCandidate>(new (std::nothrow) SyncCandidate(-1, seq_len));
  CHECK(root) << "memory alloc fail.";
  std::vector<SyncCandidate *> cands;
  auto cur = root.get();
  for (auto i = 0; i < seq_len; ++i) {
    auto sync_node = std::unique_ptr<SyncCandidate>(new (std::nothrow) SyncCandidate(i, seq_len));
    CHECK(sync_node) << "memory alloc fail.";
    sync_node->domain = CollectDomain(seq_node.child(i).child(0));
    cur->next = std::move(sync_node);
    cur = cur->next.get();
    cands.emplace_back(cur);
  }
  cur->next = std::move(root->next);  // link end and start

  for (auto cand : cands) {
    auto DetermineSyncLevel = [seq_node, dependency, context_params, domain_to_thread, domain_to_warp,
                               &cand](SyncCandidate *node) {
      auto new_dep = dependency.intersect_domain(cand->domain);
      new_dep = new_dep.intersect_range(node->domain);
      if (new_dep.is_empty()) {
        cand->InsertSyncBetween(node, Synchronization(SyncLevel::EMPTY));
      } else {
        new_dep = new_dep.intersect_params(context_params);
        if (new_dep.is_subset(new_dep.eq_at(domain_to_thread))) {
          cand->InsertSyncBetween(node, Synchronization(SyncLevel::EMPTY));
        } else if (new_dep.is_subset(new_dep.eq_at(domain_to_warp))) {
          cand->InsertSyncBetween(node, Synchronization(SyncLevel::WARP));
        } else {
          cand->InsertSyncBetween(node, Synchronization(SyncLevel::BLOCK));
        }
      }
    };
    cand->ForEachCandidateTopDown(DetermineSyncLevel);
  }

  return cur->next.get();
}

SyncCandidate *MappingOuterBand::CountSyncNumberAmongLoop(SyncCandidate *head) {
  head->ForEachCandidateTopDown([](SyncCandidate *n1) {
    auto accum_block_count = 0;
    auto accum_warp_count = 0;
    n1->ForEachCandidateTopDown([&n1, &accum_block_count, &accum_warp_count](SyncCandidate *n2) {
      auto block_count = n1->GetNumOfSyncBetween(n2, SyncLevel::BLOCK);
      auto warp_count = n1->GetNumOfSyncBetween(n2, SyncLevel::WARP);
      warp_count = std::max(warp_count - block_count, 0);

      if (accum_block_count < block_count) {
        accum_block_count = block_count;
      }
      n1->num_block_sync_to[n2] = accum_block_count;

      if (accum_warp_count < warp_count) {
        accum_warp_count = warp_count;
      }
      n1->num_warp_sync_to[n2] = accum_warp_count;
    });
  });
  return head;
}

int MappingOuterBand::GetBestSyncStartPoint(bool is_outer) {
  // When there is only one outer-band, which is the most common case, it is best to start from the beginning;
  // otherwise, we need a strategy to determine the best start point.
  return 0;
}

size_t MappingOuterBand::MapThreadHelper(isl::schedule_node &thread_root) {
  isl::schedule_node_band band_node = thread_root.as<isl::schedule_node_band>();
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";
  if (thread_cfg->bound < 1) {
    return 0;
  }

  if (!band_node) {
    LOG(WARNING) << "No permutable band to map thread.";
    return 0;
  }

  // Step 1. Determine max num dimension of threads that can be mapped.
  auto n_thread_map = CountConsecutiveCoincident(band_node);
  if (n_thread_map < 1) {
    return 0;
  }

  // Step 2. Split band node according to mapping config and coincidence of band node.
  bool has_split = false;
  if (n_thread_map > thread_cfg->bound) {
    thread_root = band_node.split(n_thread_map - thread_cfg->bound);
    thread_root = thread_root.child(0);
    has_split = true;
    n_thread_map = thread_cfg->bound;
    band_node = thread_root.as<isl::schedule_node_band>();
  }

  // split to keep nodes with coincident equals to 1
  if (n_thread_map < band_node.n_member()) {
    thread_root = band_node.split(n_thread_map);
    band_node = thread_root.as<isl::schedule_node_band>();
  } else {
    n_thread_map = static_cast<size_t>(band_node.n_member());
  }

  // Step 3. Map band under thread_root from inner dim to outer dim.
  auto after_map_pair = MapInnerDimToThreads(band_node, false, thread_cfg, scop_info_.upa_node_mapping_);
  thread_root = after_map_pair.first;
  if (has_split) {
    thread_root = thread_root.parent();
  }

  // Step 4. Do unroll if needed.
  isl::schedule_node after_fix_node = after_map_pair.second;
  thread_root = UnrollByMarkOptions(after_fix_node, scop_info_.user_config_.GetMaxUnrollLoop());

  return thread_cfg->bound;
}

isl::schedule MappingOuterBand::DoBlockMapping(const isl::schedule &sch) {
  isl::schedule_node root = sch.get_root();
  isl::schedule_node node = GetOuterBand(root);
  auto band_node = node.as<isl::schedule_node_band>();
  if (!band_node || !band_node.permutable()) {
    LOG(WARNING) << "No permutable outer band node to map block.";
    return sch;
  }

  // Step 1. Determine max num dimension of blocks that can be mapped.
  auto block_cfg = scop_info_.user_config_.GetBlockConfig();
  CHECK(block_cfg != nullptr) << "blockconfig is null";
  auto n_block_map = CountConsecutiveCoincident(band_node);
  n_block_map = std::min(block_cfg->MaxDim(), n_block_map);
  n_block_map = std::min(block_cfg->bound, n_block_map);
  if (n_block_map < 1) {
    return sch;
  }

  // Step 2. Map outerband from outer dim to inner dim.
  auto partial_schedule = band_node.get_partial_schedule();
  auto upa_list = partial_schedule.get_union_pw_aff_list();

  // Step 3. Checking extent range for mapping.
  auto domain = band_node.get_schedule().get_domain();
  isl::union_pw_aff_list range_aff_list(band_node.ctx(), static_cast<int>(upa_list.size()));
  for (size_t i = 0; i < upa_list.size(); ++i) {
    auto range = upa_list.get_at(i).intersect_domain(domain);
    range_aff_list = range_aff_list.add(range);
  }
  node = CheckMapSizeAndApplyTile(node, range_aff_list, block_cfg);
  upa_list = upa_list.drop(n_block_map, upa_list.size() - n_block_map);

  // Step 4. Create and insert mapping filter.
  node = node.insert_mark(isl::id(node.ctx(), BLOCK_MARKER));
  node = node.child(0);
  node = CreateAndInsertMapFilter(node, false, upa_list, block_cfg, scop_info_.upa_node_mapping_);

  auto final_schedule = node.get_schedule();
  return final_schedule;
}

isl::schedule MappingOuterBand::Run(isl::schedule sch) {
  auto node = sch.root().child(0);
  node = InsertContextNode(node, scop_info_);
  sch = node.schedule();

  sch = DoThreadMapping(sch);

  sch = DoBlockMapping(sch);
  return sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
