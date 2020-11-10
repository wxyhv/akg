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

#include "poly/schedule_tree_util.h"
#include "poly/schedule_pass.h"
#include "reduce_manager.h"

namespace akg {
namespace ir {
namespace poly {

isl::union_set ReduceManager::GetReduceStatements(isl::union_set domain, isl::union_map reduce_statement_map,
                                                  StatementMap all_statements) {
  isl::union_set reduce_domain = reduce_statement_map.intersect_domain(domain).domain();
  isl::union_set reduce_statements = isl::union_set::empty(reduce_domain.get_space());
  reduce_domain.foreach_set([&reduce_statements, all_statements](isl::set set) {
    isl::id id = set.get_tuple_id();

    CHECK_EQ(all_statements.count(id), 1u) << "setId is not a statement in scop" << id;
    const Node *stmt_node = all_statements.at(id);

    if (stmt_node != nullptr && stmt_node->IsInstance<Provide>()) {
      const auto provide = static_cast<const Provide *>(stmt_node);
      if (provide->value.defined()) {
        reduce_statements = reduce_statements.unite(set);
      }
    }
  });
  return reduce_statements;
}

isl::multi_union_pw_aff ReduceManager::GetCoincidentMemberRange(const isl::schedule_node &node, size_t first,
                                                                size_t num) {
  isl::schedule_node_band band_node = node.as<isl::schedule_node_band>();
  auto partial_schedule = band_node.get_partial_schedule();
  auto upa_list = partial_schedule.get_union_pw_aff_list();
  isl::space space = partial_schedule.get_space().params().add_unnamed_tuple_ui(num);
  size_t end = first + num;
  CHECK_LE(end, band_node.n_member());
  upa_list = upa_list.drop(end, band_node.n_member() - end).drop(0, first);
  return isl::multi_union_pw_aff(space, upa_list);
}

// Determine whether the first statement can be ranked before the second statement
bool ReduceManager::IsOrderStatements(isl::union_set first_statements, isl::union_set second_statements,
                                      isl::union_map dependences) {
  if (first_statements.is_empty() || second_statements.is_empty()) {
    return true;
  }
  isl::ctx ctx = dependences.ctx();
  isl::space space = isl::space(ctx, 0).add_unnamed_tuple_ui(1);
  isl::multi_val zero_first = isl::multi_val::zero(space);
  isl::multi_val one_second = zero_first.set_val(0, isl::val::one(ctx));
  auto order_statements = isl::multi_union_pw_aff(first_statements, zero_first);
  order_statements = order_statements.union_add(isl::multi_union_pw_aff(second_statements, one_second));

  auto order_dependences = dependences.lex_lt_at(order_statements).unite(dependences.eq_at(order_statements));
  return dependences.is_subset(order_dependences);
}

isl::schedule_node ReduceManager::OrderStatements(const isl::schedule_node &node, isl::union_set before,
                                                  isl::union_set after) {
  isl::union_set middle = CollectDomain(node);
  isl::schedule_node order_node = node;
  isl::union_set_list filter_list;
  size_t depth = (before.is_empty() && !after.is_empty()) ? 0 : 1;
  auto AddToFilterList = [this, &filter_list](const isl::set &s) -> void {
    filter_list = filter_list.add(isl::union_set(s));
  };
  if (!before.is_empty() && after.is_empty()) {
    middle = middle.subtract(before);
    filter_list = isl::union_set_list(before);
    middle.foreach_set(AddToFilterList);
  } else if (before.is_empty() && !after.is_empty()) {
    middle = middle.subtract(after);
    middle.foreach_set(AddToFilterList);
    filter_list = filter_list.add(after);
  } else {
    middle = middle.subtract(before).subtract(after);
    filter_list = isl::union_set_list(before);
    middle.foreach_set(AddToFilterList);
    filter_list = filter_list.add(after);
  }

  order_node = order_node.insert_sequence(filter_list);
  order_node = order_node.insert_mark(INSERT_SYNC);
  order_node = order_node.child(0).child(depth);

  return order_node;
}

// Separate the reduce statement from other statements
bool ReduceManager::SplitReduceStatements(isl::schedule_node &node, isl::union_set reduce_statements,
                                          isl::union_map dependences) {
  auto domain = CollectDomain(node);
  auto no_reduce_statements = domain.subtract(reduce_statements);
  if (no_reduce_statements.is_empty()) {
    return true;
  }

  auto prefix = ShortScheduleMupaImpl(node.root(), node.root(), node.parent());
  isl::union_map active_dependences = dependences.intersect_domain(domain).intersect_range(domain).eq_at(prefix);

  isl::union_set depend_reduce =
    active_dependences.intersect_domain(reduce_statements).intersect_range(no_reduce_statements).range();
  isl::union_set no_depend_reduce = no_reduce_statements.subtract(depend_reduce);

  isl::union_set other_depend_reduce = domain.subtract(depend_reduce);
  isl::union_set other_no_depend_reduce = domain.subtract(no_depend_reduce);
  if (!IsOrderStatements(no_depend_reduce, other_no_depend_reduce, dependences) ||
      !IsOrderStatements(other_depend_reduce, depend_reduce, dependences)) {
    return false;
  }

  // Statements that do not depend on the reduce statement are ordered before.
  // Statements that depend on the reduce statement are ordered after.
  if (no_depend_reduce.is_empty() && depend_reduce.is_empty()) {
    return false;
  } else {
    node = OrderStatements(node, no_depend_reduce, depend_reduce);
  }

  return true;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
