cmake_minimum_required(VERSION 2.8)

project(glia)

option(GLIA_3D "Process 3D images other than 2D." OFF)
option(GLIA_MT "Use multi-threading." OFF)
option(GLIA_BUILD_HMT "Build HMT/CCHMT module." ON)
option(GLIA_BUILD_SSHMT "Build SSHMT module. (Requires HMT module.)" ON)
option(GLIA_BUILD_LINK3D "Build 3D linking module. (Requires GLIA_3D=OFF.)" ON)
option(GLIA_BUILD_GADGET "Build gadget module." ON)
option(GLIA_BUILD_ML_RF "Build random forest module. (Requires 3rd party random forest code in place.)" OFF)

if(GLIA_BUILD_ML_RF)
  set(RF_SRC_DIR "" CACHE FILEPATH "Random forest source code directory")
endif(GLIA_BUILD_ML_RF)

if(GLIA_3D)
  add_definitions(-DGLIA_3D)
endif(GLIA_3D)

if(GLIA_MT)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DGLIA_MT)
  endif(OPENMP_FOUND)
endif(GLIA_MT)

include_directories(.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set(Eigen_INCLUDE_DIR "" CACHE FILEPATH "Eigen source directory")
include_directories(${Eigen_INCLUDE_DIR})

set(Boost_INCLUDE_DIR "" CACHE FILEPATH "Boost include directory")
set(Boost_LIBRARY_DIR "" CACHE FILEPATH "Boost library directory")
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# HMT/CCHMT
if(GLIA_BUILD_HMT)
  option(GLIA_HMT_HIST_FEAT "Use histograms as features." OFF)
  option(GLIA_HMT_MEDIAN_FEAT "Use medians as features." OFF)
  option(GLIA_HMT_USE_RF "Use random forest for merge tree generation. (Require 3rd party random forest code in place)" OFF)

  if(GLIA_HMT_HIST_FEAT)
    add_definitions(-DGLIA_USE_HISTOGRAM_AS_FEATS)
  endif(GLIA_HMT_HIST_FEAT)

  if(GLIA_HMT_MEDIAN_FEAT)
    add_definitions(-DGLIA_USE_MEDIAN_AS_FEATS)
  endif(GLIA_HMT_MEDIAN_FEAT)

  add_executable(watershed gadget/main_watershed.cxx)
  target_link_libraries(watershed ${ITK_LIBRARIES} boost_program_options)

  add_executable(pre_merge gadget/main_pre_merge.cxx)
  target_link_libraries(pre_merge ${ITK_LIBRARIES} boost_program_options)

  add_executable(merge_order_pb hmt/main_merge_order_pb.cxx)
  target_link_libraries(merge_order_pb ${ITK_LIBRARIES} boost_program_options)

  add_executable(bc_feat hmt/main_bc_feat.cxx)
  target_link_libraries(bc_feat ${ITK_LIBRARIES} boost_program_options)

  add_executable(bc_label_vi hmt/main_bc_label_vi.cxx)
  target_link_libraries(bc_label_vi ${ITK_LIBRARIES} boost_program_options)

  add_executable(bc_label_ri hmt/main_bc_label_ri.cxx)
  target_link_libraries(bc_label_ri ${ITK_LIBRARIES} boost_program_options)

  add_executable(segment_greedy hmt/main_segment_greedy.cxx)
  target_link_libraries(segment_greedy ${ITK_LIBRARIES} boost_program_options)

  add_executable(segment_ccm hmt/main_segment_ccm.cxx)
  target_link_libraries(segment_ccm ${ITK_LIBRARIES} boost_program_options)

  if(GLIA_HMT_USE_RF)
    if(NOT DEFINED RF_SRC_DIR)
      set(RF_SRC_DIR "" CACHE FILEPATH "Random forest source code directory")
    endif(NOT DEFINED RF_SRC_DIR)

    enable_language(Fortran)
    set(CMAKE_Fortran_FLAGS "-fpic -m64")

    add_executable(merge_order_bc
      hmt/main_merge_order_bc.cxx
      ml/rf/ml_rf_model.cxx
      ${RF_SRC_DIR}/classRF.cpp
      ${RF_SRC_DIR}/classTree.cpp
      ${RF_SRC_DIR}/cokus.cpp
      ${RF_SRC_DIR}/qsort.c
      ${RF_SRC_DIR}/rfsub.f
      ${RF_SRC_DIR}/rfutils.cpp)
    target_link_libraries(merge_order_bc ${ITK_LIBRARIES} boost_program_options)
  endif(GLIA_HMT_USE_RF)
  
  # SSHMT
  if(GLIA_BUILD_SSHMT)
    add_executable(bc_rc_label_ji hmt/main_bc_rc_label_ji.cxx)
    target_link_libraries(bc_rc_label_ji ${ITK_LIBRARIES} boost_program_options)
  
    add_executable(train_sshmt_logsig sshmt/main_train_sshmt_logsig.cxx)
    target_link_libraries(train_sshmt_logsig boost_program_options)

    add_executable(train_sshmt_mlp sshmt/main_train_sshmt_mlp.cxx)
    target_link_libraries(train_sshmt_mlp boost_program_options)
  
    add_executable(pred_logsig sshmt/main_pred_logsig.cxx)
    target_link_libraries(pred_logsig boost_program_options)

    add_executable(pred_mlp sshmt/main_pred_mlp.cxx)
    target_link_libraries(pred_mlp boost_program_options)
  endif(GLIA_BUILD_SSHMT)
endif(GLIA_BUILD_HMT)

# 3D Linking
if((NOT GLIA_3D) AND GLIA_BUILD_LINK3D)
  add_executable(gen_region_pairs gadget/main_gen_region_pairs)
  target_link_libraries(gen_region_pairs ${ITK_LIBRARIES} boost_program_options)

  add_executable(sc_feat hmt/main_sc_feat.cxx)
  target_link_libraries(sc_feat ${ITK_LIBRARIES} boost_program_options)

  add_executable(sc_label_ri hmt/main_sc_label_ri.cxx)
  target_link_libraries(sc_label_ri ${ITK_LIBRARIES} boost_program_options)

  add_executable(link_by_threshold gadget/main_link_by_threshold)
  target_link_libraries(link_by_threshold boost_program_options)

  add_executable(group_region_profiles gadget/main_group_region_profiles)
  target_link_libraries(group_region_profiles ${ITK_LIBRARIES} boost_program_options)
endif((NOT GLIA_3D) AND GLIA_BUILD_LINK3D)

# Gadget
if(GLIA_BUILD_GADGET)
  add_executable(threshold_image gadget/main_threshold_image.cxx)
  target_link_libraries(threshold_image ${ITK_LIBRARIES} boost_program_options)

  add_executable(labelcc_image gadget/main_labelcc_image.cxx)
  target_link_libraries(labelcc_image ${ITK_LIBRARIES} boost_program_options)

  add_executable(labelscc_image gadget/main_labelscc_image.cxx)
  target_link_libraries(labelscc_image ${ITK_LIBRARIES} boost_program_options)

  add_executable(labelicc_image gadget/main_labelicc_image.cxx)
  target_link_libraries(labelicc_image ${ITK_LIBRARIES} boost_program_options)

  add_executable(relabel_image gadget/main_relabel_image.cxx)
  target_link_libraries(relabel_image ${ITK_LIBRARIES} boost_program_options)

  add_executable(resample_image gadget/main_resample_image.cxx)
  target_link_libraries(resample_image ${ITK_LIBRARIES} boost_program_options)

  add_executable(maxpool_image gadget/main_maxpool_image.cxx)
  target_link_libraries(maxpool_image ${ITK_LIBRARIES} boost_program_options)

  add_executable(blur_image gadget/main_blur_image.cxx)
  target_link_libraries(blur_image ${ITK_LIBRARIES} boost_program_options)

  add_executable(crop_image gadget/main_crop_image.cxx)
  target_link_libraries(crop_image ${ITK_LIBRARIES} boost_program_options)

  add_executable(acc_images gadget/main_acc_images.cxx)
  target_link_libraries(acc_images ${ITK_LIBRARIES} boost_program_options)

  add_executable(image_compression gadget/main_image_compression.cxx)
  target_link_libraries(image_compression ${ITK_LIBRARIES} boost_program_options)

  add_executable(apply_merges gadget/main_apply_merges.cxx)
  target_link_libraries(apply_merges ${ITK_LIBRARIES} boost_program_options)

  add_executable(label_image_stats gadget/main_label_image_stats.cxx)
  target_link_libraries(label_image_stats ${ITK_LIBRARIES} boost_program_options)

  add_executable(eval_vi gadget/main_eval_vi.cxx)
  target_link_libraries(eval_vi ${ITK_LIBRARIES} boost_program_options)

  add_executable(eval_ri gadget/main_eval_ri.cxx)
  target_link_libraries(eval_ri ${ITK_LIBRARIES} boost_program_options)

  add_executable(eval_ri_threshold gadget/main_eval_ri_threshold.cxx)
  target_link_libraries(eval_ri_threshold ${ITK_LIBRARIES} boost_program_options)

  add_executable(normalize_sample gadget/main_normalize_sample.cxx)
  target_link_libraries(normalize_sample boost_program_options)

  add_executable(unique_sample gadget/main_unique_sample.cxx)
  target_link_libraries(unique_sample boost_program_options)

  add_executable(distribute_samples gadget/main_distribute_samples.cxx)
  target_link_libraries(distribute_samples boost_program_options)

  add_executable(distribute_label_images gadget/main_distribute_label_images.cxx)
  target_link_libraries(distribute_label_images ${ITK_LIBRARIES} boost_program_options)

  add_executable(select_hard_samples gadget/main_select_hard_samples.cxx)
  target_link_libraries(select_hard_samples boost_program_options)

  add_executable(gen_image_patches gadget/main_gen_image_patches.cxx)
  target_link_libraries(gen_image_patches ${ITK_LIBRARIES} boost_program_options)

  add_executable(eval_init_seg gadget/main_eval_init_seg.cxx)
  target_link_libraries(eval_init_seg ${ITK_LIBRARIES} boost_program_options)

  add_executable(seg_stats gadget/main_seg_stats.cxx)
  target_link_libraries(seg_stats ${ITK_LIBRARIES} boost_program_options)

  add_executable(image_vol_to_slices gadget/main_image_vol_to_slices.cxx)
  target_link_libraries(image_vol_to_slices ${ITK_LIBRARIES} boost_program_options)

  add_executable(match_seg_to_truth gadget/main_match_seg_to_truth.cxx)
  target_link_libraries(match_seg_to_truth ${ITK_LIBRARIES} boost_program_options)

  add_executable(match_truth_to_seg gadget/main_match_truth_to_seg.cxx)
  target_link_libraries(match_truth_to_seg ${ITK_LIBRARIES} boost_program_options)

  if(NOT GLIA_3D)
    add_executable(resample_rgb_image gadget/main_resample_rgb_image.cxx)
    target_link_libraries(resample_rgb_image ${ITK_LIBRARIES} boost_program_options)

    add_executable(boundary_image_2d gadget/main_boundary_image_2d.cxx)
    target_link_libraries(boundary_image_2d ${ITK_LIBRARIES} boost_program_options)

    add_executable(overlay_image gadget/main_overlay_image.cxx)
    target_link_libraries(overlay_image ${ITK_LIBRARIES} boost_program_options)
  endif(NOT GLIA_3D)
endif(GLIA_BUILD_GADGET)

# ML/RF
if(GLIA_BUILD_ML_RF)
  enable_language(Fortran)
  set(CMAKE_Fortran_FLAGS "-fpic -m64")

  add_executable(train_rf
    ml/rf/main_train_rf.cxx
    ml/rf/ml_rf_train.cxx
    ml/rf/ml_rf_model.cxx
    ml/rf/ml_rf_util.cxx
    ${RF_SRC_DIR}/classRF.cpp
    ${RF_SRC_DIR}/classTree.cpp
    ${RF_SRC_DIR}/cokus.cpp
    ${RF_SRC_DIR}/qsort.c
    ${RF_SRC_DIR}/rfsub.f
    ${RF_SRC_DIR}/rfutils.cpp)
  target_link_libraries(train_rf boost_program_options)

  add_executable(pred_rf
    ml/rf/main_pred_rf.cxx
    ml/rf/ml_rf_model.cxx
    ${RF_SRC_DIR}/classRF.cpp
    ${RF_SRC_DIR}/classTree.cpp
    ${RF_SRC_DIR}/cokus.cpp
    ${RF_SRC_DIR}/qsort.c
    ${RF_SRC_DIR}/rfsub.f
    ${RF_SRC_DIR}/rfutils.cpp)
  target_link_libraries(pred_rf boost_program_options)
endif(GLIA_BUILD_ML_RF)
